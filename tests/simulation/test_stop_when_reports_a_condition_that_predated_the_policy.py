"""A ``stop_when`` clause the scene already satisfies is reported, not read as an achievement.

``stop_when`` is evaluated only AFTER an applied action, so a clause the
initial state already satisfies fires on the first step whatever the policy
commands. The rollout then reports ``stopped_reason="predicate"`` after one
step - the field an agent reads to tell "the world reached the goal state" from
"the step budget ran out" - and nothing in the payload separated it from a
rollout that drove the world there.

This is the mirror of the corruption the pre-rollout entity probe already
refuses: a typo'd body name degrades to a constant ``False``, burns the whole
budget and reports ``stopped_reason="budget"``, indistinguishable from an
honest miss. The reset-true direction is indistinguishable from an honest
achievement, and no compile-time or name-resolution check can see it because it
is a fact about the initial state rather than about the clause.

The cost lands hardest on a collection loop, where ``stop_when`` is documented
as a per-episode success gate: every episode ends after one step, so the
recorded dataset holds one frame per episode, each tagged as having reached the
condition.

Reported rather than refused, the posture ``episodes_successful_at_reset``
already takes on the two evaluation routes: domain randomisation legitimately
draws an initial state that satisfies a clause on some episodes, so every
reported figure is left as measured.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import stop_when_true_at_reset_warning
from strands_robots.tools.run_policy import run_policy as run_policy_tool

# A cube resting on the floor at z=0.025: ``body_above_z`` at 0.01 is already
# true, at 5.0 can never become true, and a cube spawned in the air crosses 0.5
# mid-rollout as it falls - the three cases a caller cannot otherwise tell
# apart from the payload.
_RESTING = {"predicate": "body_above_z", "body": "cube", "z": 0.01}
_UNREACHABLE = {"predicate": "body_above_z", "body": "cube", "z": 5.0}


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


@pytest.fixture
def sim_resting_cube():
    """so100 + a cube already at rest on the floor (z = half its 0.05 extent)."""
    s = Simulation(tool_name="stop_when_reset_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    assert (
        s.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.025], size=[0.05, 0.05, 0.05])["status"]
        == "success"
    )
    yield s
    s.cleanup()


@pytest.fixture
def sim_falling_cube():
    """so100 + a cube spawned in the air, so a z=0.5 clause fires MID-rollout."""
    s = Simulation(tool_name="stop_when_reset_control", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    assert (
        s.add_object(name="cube", shape="box", position=[0.4, 0.0, 1.0], size=[0.05, 0.05, 0.05])["status"] == "success"
    )
    yield s
    s.cleanup()


def _run(sim, **kw: Any) -> dict[str, Any]:
    kw.setdefault("n_steps", 12)
    return sim.run_policy(
        robot_name="alice",
        policy_provider="mock",
        instruction="pick up the cube",
        control_frequency=50.0,
        fast_mode=True,
        **kw,
    )


class TestRolloutReportsIt:
    def test_a_clause_the_initial_state_satisfies_is_flagged_and_qualified(self, sim_resting_cube):
        """The reporting the payload lacked: the flag, the text, and the warning."""
        payload = _json_block(_run(sim_resting_cube, stop_when=_RESTING))
        assert payload["stopped_reason"] == "predicate"
        assert payload["steps_used"] == 1
        assert payload["stop_when_true_at_reset"] is True
        assert payload["stop_when_reset_warning"] == stop_when_true_at_reset_warning(surface="run_policy")

    def test_the_warning_reaches_the_human_readable_text(self, sim_resting_cube):
        assert "stop_when_true_at_reset" in _text(_run(sim_resting_cube, stop_when=_RESTING))

    @pytest.mark.parametrize(
        "clause_name",
        ["unreachable", "none"],
        ids=["a clause that never fires", "no clause at all"],
    )
    def test_the_flag_is_present_and_false_when_nothing_predated_the_policy(self, sim_resting_cube, clause_name):
        """Always present so callers can rely on the key - a missing key reads as
        absent evidence, which is what this flag exists to distinguish."""
        clause = _UNREACHABLE if clause_name == "unreachable" else None
        payload = _json_block(_run(sim_resting_cube, stop_when=clause))
        assert payload["stopped_reason"] == "budget"
        assert payload["stop_when_true_at_reset"] is False
        assert payload["stop_when_reset_warning"] is None

    def test_a_genuine_mid_rollout_achievement_is_not_flagged(self, sim_falling_cube):
        """The control that gives the flag its meaning: a clause fired by the world
        CHANGING rather than by the world starting there. The cube spawns at
        z=1.0, so ``body_below_z`` at 0.5 is false at reset and becomes true as
        it falls - the same ``stopped_reason="predicate"``, honestly earned.
        """
        payload = _json_block(
            _run(
                sim_falling_cube,
                n_steps=40,
                stop_when={"predicate": "body_below_z", "body": "cube", "z": 0.5},
            )
        )
        assert payload["stopped_reason"] == "predicate"
        assert payload["steps_used"] > 1
        assert payload["stop_when_true_at_reset"] is False
        assert payload["stop_when_reset_warning"] is None

    def test_the_reset_sample_does_not_capture_a_raising_clause(self, sim_resting_cube):
        """The pre-rollout sample is diagnostic, so it is NOT fatal on a raise -
        unlike the per-step call. A raising clause must still be attributed to
        the step it raised on rather than to the probe."""

        def boom(_sim: Any) -> bool:
            raise RuntimeError("clause exploded")

        result = _run(sim_resting_cube, stop_when=boom)
        assert result["status"] == "error"
        assert "at step" in _text(result)


class TestCollectionLoopReportsIt:
    """The surface where the harm bites: the tool's per-episode success gate."""

    class _Sim:
        """Minimal stand-in whose rollouts report a reset-true clause.

        Emits both fields the real payload carries: the tool propagates the
        warning text from the rollout that reported it rather than re-deriving
        it, so a double that omitted it would pin the wrong contract.
        """

        def __init__(self, at_reset: bool) -> None:
            self.at_reset = at_reset

        def run_policy(self, **_kw: Any) -> dict[str, Any]:
            return {
                "status": "success",
                "content": [
                    {"text": "rollout"},
                    {
                        "json": {
                            "stopped_reason": "predicate" if self.at_reset else "budget",
                            "steps_used": 1 if self.at_reset else 12,
                            "stop_when_true_at_reset": self.at_reset,
                            "stop_when_reset_warning": (
                                stop_when_true_at_reset_warning(surface="run_policy") if self.at_reset else None
                            ),
                        }
                    },
                ],
            }

    @staticmethod
    def _tool() -> Any:
        for attr in ("_tool_func", "original_function", "__wrapped__", "func"):
            if callable(target := getattr(run_policy_tool, attr, None)):
                return target
        return run_policy_tool

    @pytest.mark.parametrize("at_reset", [True, False], ids=["predated the policy", "honest"])
    def test_every_episode_carries_the_flag_and_the_loop_aggregates_it(self, at_reset):
        payload = _json_block(
            self._tool()(
                self._Sim(at_reset),
                n_episodes=3,
                n_steps=12,
                stop_when={"predicate": "body_above_z", "body": "cube", "z": 0.01},
            )
        )
        assert [e["stop_when_true_at_reset"] for e in payload["episodes"]] == [at_reset] * 3
        assert payload["episodes_stop_when_true_at_reset"] == (3 if at_reset else 0)
        assert payload["stop_when_reset_warning"] == (
            stop_when_true_at_reset_warning(surface="run_policy") if at_reset else None
        )

    def test_it_qualifies_the_collection_without_failing_it(self):
        """Not a ``warnings`` entry: that flips ``status`` to "error", and a
        randomised initial state legitimately satisfies a clause on some draws."""
        result = self._tool()(
            self._Sim(True),
            n_episodes=2,
            n_steps=12,
            stop_when={"predicate": "body_above_z", "body": "cube", "z": 0.01},
        )
        assert result["status"] == "success"
        assert _json_block(result)["warnings"] == []
        assert "stop_when_true_at_reset=2/2" in _text(result)


def test_the_warning_names_the_field_that_carries_it() -> None:
    """A warning whose remedy names no reportable field leaves the reader nowhere."""
    warning = stop_when_true_at_reset_warning(surface="run_policy")
    assert "run_policy" in warning
    assert "stop_when_true_at_reset" in warning
    assert "stopped_reason='predicate'" in warning
