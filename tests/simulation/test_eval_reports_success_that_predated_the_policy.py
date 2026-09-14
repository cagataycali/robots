"""An episode already successful at reset is counted, not silently averaged in.

Both evaluation routes sample the success criterion only AFTER an applied action.
An episode whose criterion already holds at reset therefore succeeds on its first
step whatever the policy commands, and ``success_rate`` reports a hard ``1.0`` for
it - the mirror of the ``success_measured=False`` case, where a missing criterion
reports a hard ``0.0`` and is warned about in exactly those terms. The route this
package already refuses,
:func:`~strands_robots.simulation.policy_runner.uncommanded_eval_error`, is the
same harm reached the other way: there the policy never commands, here the
criterion never needed it, and either way "the reported metrics describe the
scene's initial state rather than the policy".

It is a real configuration rather than a contrived one. The threshold only has to
sit on the wrong side of the initial state, which is what a lift predicate does
whenever the object rests on a support and the height is measured from the floor -
the spec below asks the cube to get above 1cm while it already rests at 2.5cm.

Reported rather than refused, and every figure is left as measured: domain
randomisation draws initial states per episode, so a partial count is a fact about
those draws rather than a broken spec, and the count is what distinguishes them.
That is the posture ``_warn_unresolved`` states for a criterion degenerating to a
constant - surface the corruption without changing a returned value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.simulation.policy_runner import success_at_reset_warning

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import _BENCHMARK_REGISTRY  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A one-joint arm plus a cube resting on the floor. The cube's half-extent puts
#: its body origin at :data:`_CUBE_REST_Z`, which is the only scene fact the specs
#: below are written against.
_SCENE_XML = """
<mujoco model="reset_success_probe">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
    <body name="cube" pos="0.5 0 0.025">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="10"/>
  </actuator>
</mujoco>
"""

#: The same arm with nothing for anything to touch: no floor, and the cube pinned
#: in the air. ``success_fn="contact"`` is then false at reset and stays false, so
#: the same criterion is scored honestly - the control for the legacy route.
_SCENE_XML_UNTOUCHED = """
<mujoco model="reset_success_probe_untouched">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
    <body name="cube" pos="0.9 0 0.5">
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="10"/>
  </actuator>
</mujoco>
"""

#: Height the cube's origin already sits at, resting on the floor.
_CUBE_REST_Z = 0.025

#: A lift threshold BELOW where the cube already rests - satisfied at reset.
_Z_ALREADY_SATISFIED = 0.01

#: A lift threshold the probe scene never reaches, so the same rollout is scored
#: honestly. The control that keeps the count from being a constant.
_Z_NEVER_SATISFIED = 0.20

_N_EPISODES = 3
_MAX_STEPS = 4


@pytest.fixture(autouse=True)
def _clean_registry():
    """Leave the module-global benchmark registry as it was found."""
    before = dict(_BENCHMARK_REGISTRY)
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(before)


@pytest.fixture
def sim():
    engine = Simulation()
    yield engine
    engine.destroy()


def _world(tmp_path: Path, sim, xml: str, name: str) -> str:
    # No implicit ground plane: each scene below declares exactly the surfaces it
    # wants, so "nothing is in contact" is a property of the scene rather than of
    # what ``create_world`` happens to add.
    sim.create_world(ground_plane=False)
    scene = tmp_path / f"{name}.xml"
    scene.write_text(xml)
    sim.add_robot("arm1", urdf_path=str(scene))
    return "arm1"


@pytest.fixture
def arm(tmp_path: Path, sim) -> str:
    """The arm beside a cube already resting on the floor."""
    return _world(tmp_path, sim, _SCENE_XML, "probe")


@pytest.fixture
def arm_untouched(tmp_path: Path, sim) -> str:
    """The arm with nothing in contact, at reset or ever."""
    return _world(tmp_path, sim, _SCENE_XML_UNTOUCHED, "probe_untouched")


def _lift_task(tmp_path: Path, sim, arm: str, z: float, name: str) -> str:
    """Register the canonical lift benchmark with its threshold at ``z``."""
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": name,
                "default_robot": arm,
                "supported_robots": [],
                "max_steps": _MAX_STEPS,
                "instruction": "lift the cube",
                "success": {"all": [{"predicate": "body_above_z", "body": "cube", "z": z}]},
            }
        )
    )
    result = sim.register_benchmark_from_file(name, str(spec_path))
    assert result.get("status") != "error", result
    return name


def _metrics(result: dict) -> dict:
    assert result.get("status") != "error", result
    return next(c["json"] for c in result["content"] if "json" in c)


def _spec_eval(sim, task: str) -> dict:
    return _metrics(sim.evaluate_benchmark(task, policy_provider="mock", n_episodes=_N_EPISODES, seed=0))


def _legacy_eval(sim, arm: str) -> dict:
    """The ``success_fn`` route, driven by the shipped ``"contact"`` criterion.

    A criterion of "something is touching" is satisfied at reset by any scene whose
    object already rests on a support, which is most manipulation scenes.
    """
    return _metrics(
        sim.eval_policy(
            robot_name=arm,
            policy_provider="mock",
            success_fn="contact",
            n_episodes=_N_EPISODES,
            max_steps=_MAX_STEPS,
        )
    )


class TestTheCountIsReportedOnBothEvaluationRoutes:
    """``uncommanded_eval_error`` covers both routes; so does this."""

    def test_an_already_satisfied_spec_criterion_is_counted(self, tmp_path, sim, arm) -> None:
        metrics = _spec_eval(sim, _lift_task(tmp_path, sim, arm, _Z_ALREADY_SATISFIED, "reset-true"))
        self._assert_counted(metrics)

    def test_an_already_satisfied_success_fn_is_counted(self, sim, arm) -> None:
        self._assert_counted(_legacy_eval(sim, arm))

    @staticmethod
    def _assert_counted(metrics: dict) -> None:
        assert metrics["episodes_successful_at_reset"] == metrics["episodes_completed"], (
            f"the cube rests at {_CUBE_REST_Z}m and the criterion asks for {_Z_ALREADY_SATISFIED}m, so every "
            f"episode was successful before the policy acted, but the payload counts "
            f"{metrics['episodes_successful_at_reset']} of {metrics['episodes_completed']}"
        )
        assert metrics["reset_success_warning"] is not None, (
            "success_rate is a hard 1.0 here regardless of the policy and the payload says nothing about it"
        )

    def test_a_spec_criterion_the_scene_does_not_satisfy_is_not_counted(self, tmp_path, sim, arm) -> None:
        """The control: the count is a measurement, not a constant."""
        self._assert_not_counted(_spec_eval(sim, _lift_task(tmp_path, sim, arm, _Z_NEVER_SATISFIED, "honest")))

    def test_a_success_fn_the_scene_does_not_satisfy_is_not_counted(self, sim, arm_untouched) -> None:
        """The same control on the legacy route: nothing touches, at reset or ever."""
        self._assert_not_counted(_legacy_eval(sim, arm_untouched))

    @staticmethod
    def _assert_not_counted(metrics: dict) -> None:
        assert metrics["episodes_successful_at_reset"] == 0, (
            "the criterion is not satisfied by this scene at reset, so no episode was decided before the "
            f"policy acted - counted {metrics['episodes_successful_at_reset']} of "
            f"{metrics['episodes_completed']}"
        )
        assert metrics["reset_success_warning"] is None, "an honestly-scored evaluation is not qualified"
        assert metrics["success_rate"] == 0.0, "the probe scene cannot reach this threshold"

    def test_each_episode_record_carries_its_own_verdict(self, tmp_path, sim, arm) -> None:
        """Per-episode, because a randomised initial state decides it per episode."""
        metrics = _spec_eval(sim, _lift_task(tmp_path, sim, arm, _Z_ALREADY_SATISFIED, "reset-true"))
        flags = [row["success_at_reset"] for row in metrics["episodes"]]
        assert flags == [True] * metrics["episodes_completed"], f"per-episode verdicts read {flags}"


class TestTheEvaluationIsQualifiedRatherThanRefusedOrRewritten:
    """The posture claim: a reader gains a caveat and loses no measurement."""

    def test_the_success_rate_is_left_as_measured(self, tmp_path, sim, arm) -> None:
        metrics = _spec_eval(sim, _lift_task(tmp_path, sim, arm, _Z_ALREADY_SATISFIED, "reset-true"))
        assert metrics["success_rate"] == 1.0, "the rate is reported as measured, not corrected downward"
        assert metrics["n_success"] == metrics["episodes_completed"]

    def test_the_evaluation_still_ran(self, tmp_path, sim, arm) -> None:
        task = _lift_task(tmp_path, sim, arm, _Z_ALREADY_SATISFIED, "reset-true")
        result = sim.evaluate_benchmark(task, policy_provider="mock", n_episodes=_N_EPISODES, seed=0)
        assert result["status"] == "success", "a partial count is a fact about the draws, not an error"

    def test_the_warning_reaches_the_human_readable_block(self, tmp_path, sim, arm) -> None:
        task = _lift_task(tmp_path, sim, arm, _Z_ALREADY_SATISFIED, "reset-true")
        result = sim.evaluate_benchmark(task, policy_provider="mock", n_episodes=_N_EPISODES, seed=0)
        text = "\n".join(c["text"] for c in result["content"] if "text" in c)
        assert "at reset" in text, f"the text block reports 100% success without the caveat:\n{text}"


class TestSuccessAtResetWarning:
    """The message states the count, the scope of the doubt, and the remedy."""

    def test_no_affected_episode_is_not_qualified(self) -> None:
        assert (
            success_at_reset_warning(surface="eval_policy", episodes_completed=3, episodes_successful_at_reset=0)
            is None
        )

    def test_an_evaluation_that_completed_nothing_is_not_qualified(self) -> None:
        assert (
            success_at_reset_warning(surface="eval_policy", episodes_completed=0, episodes_successful_at_reset=0)
            is None
        )

    @pytest.mark.parametrize(
        ("affected", "completed", "expected_scope"),
        [(3, 3, "the reported success_rate"), (1, 3, "that part of the reported success_rate")],
        ids=["every-episode", "some-episodes"],
    )
    def test_the_scope_of_the_doubt_tracks_the_count(self, affected, completed, expected_scope) -> None:
        message = success_at_reset_warning(
            surface="evaluate_benchmark",
            episodes_completed=completed,
            episodes_successful_at_reset=affected,
        )
        assert message is not None
        assert f"{affected} of {completed} episode(s)" in message, message
        assert expected_scope in message, message
        assert "evaluate_benchmark" in message, "the caller is named so the message points at their call"
        assert "episodes_successful_at_reset" in message, "the remedy names the field that carries the count"
