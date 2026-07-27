"""``run_policy(stop_when=...)`` ends a rollout on a world state, not only on a budget.

A rollout used as a retryable primitive needs a semantic horizon: "run until the
cube is lifted", not "run 300 steps and hope". These tests pin the contract of
the ``stop_when`` predicate clause and of the ``stopped_reason`` telemetry that
tells a caller WHY a rollout ended - a goal reached, a budget exhausted, a
cancellation, or a failure. ``stopped_early`` alone conflates the last three.

The falling cube is the oracle throughout: free fall from a known height is
deterministic, so a condition on its height fires at a step count that does not
depend on what the driving policy does to the arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import CooperativeStop, PolicyRunner

CUBE_START_Z = 0.6
# Free fall from CUBE_START_Z crosses this height in ~0.14 s, i.e. ~7 control
# steps at 50 Hz - comfortably inside the 200-step budget the tests request, so
# an early return is unambiguous.
CUBE_STOP_Z = 0.4
BUDGET = 200


@pytest.fixture
def sim():
    """MuJoCo world with an arm to drive and a cube in free fall."""
    s = Simulation(tool_name="stop_when_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    s.add_object(
        name="cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.4, 0.0, CUBE_START_Z],
        mass=0.1,
    )
    yield s
    s.cleanup()


def rollout_json(result: dict[str, Any]) -> dict[str, Any]:
    """Return the agent-consumable json block of a rollout result."""
    for block in result["content"]:
        if "json" in block:
            return dict(block["json"])
    raise AssertionError(f"result carries no json block: {result}")


def cube_height(sim: Simulation) -> float:
    """Cube z read from the live engine (the same source the predicate reads)."""
    return float(rollout_json(sim.get_body_state("cube"))["position"][2])


def cube_below_stop_z(sim: Simulation) -> bool:
    return cube_height(sim) < CUBE_STOP_Z


class TestStopWhenEndsTheRollout:
    def test_predicate_clause_returns_early_and_reports_the_predicate(self, sim):
        """The rollout stops the step the world satisfies the clause."""
        result = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            fast_mode=True,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
        )
        assert result["status"] == "success"
        payload = rollout_json(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["stopped_early"] is True
        # n_steps is the steps actually executed, so it is the caller's
        # "how long did this take" number - well below the requested budget.
        assert 0 < payload["n_steps"] < BUDGET
        # And the world really is in the state the clause named.
        assert cube_below_stop_z(sim)

    def test_group_clause_requires_every_member_of_all(self, sim):
        """An ``all`` group only fires once every listed predicate holds."""
        unreachable = sim.run_policy(
            robot_name="alice",
            n_steps=20,
            fast_mode=True,
            stop_when={
                "all": [
                    {"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
                    {"predicate": "body_below_z", "body": "cube", "z": -5.0},
                ]
            },
        )
        assert rollout_json(unreachable)["stopped_reason"] == "budget"

        sim.reset()
        satisfiable = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            fast_mode=True,
            stop_when={
                "any": [
                    {"predicate": "body_below_z", "body": "cube", "z": -5.0},
                    {"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
                ]
            },
        )
        assert rollout_json(satisfiable)["stopped_reason"] == "predicate"

    def test_callable_condition_is_accepted_and_receives_the_engine(self, sim):
        """Programmatic callers may pass a ``(sim) -> bool`` instead of a clause."""
        seen: list[Any] = []

        def condition(engine: Any) -> bool:
            seen.append(engine)
            return cube_below_stop_z(engine)

        result = sim.run_policy(robot_name="alice", n_steps=BUDGET, fast_mode=True, stop_when=condition)
        payload = rollout_json(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["n_steps"] < BUDGET
        # The engine itself is handed to the condition, so it can query any part
        # of the world - not just whatever the observation dict happens to carry.
        assert seen and all(engine is sim for engine in seen)

    def test_condition_that_cannot_be_evaluated_fails_the_rollout(self, sim):
        """A raising condition is an error, never a silently-unmet condition."""

        def broken(_engine: Any) -> bool:
            raise RuntimeError("cannot read the world")

        result = sim.run_policy(robot_name="alice", n_steps=20, fast_mode=True, stop_when=broken)
        assert result["status"] == "error"
        assert "cannot read the world" in result["content"][0]["text"]
        assert rollout_json(result)["stopped_reason"] == "error"


class TestStoppedReason:
    def test_exhausted_budget_reports_budget(self, sim):
        """A condition that never holds runs the full horizon and says so."""
        result = sim.run_policy(
            robot_name="alice",
            n_steps=10,
            fast_mode=True,
            stop_when={"predicate": "body_above_z", "body": "cube", "z": 99.0},
        )
        payload = rollout_json(result)
        assert payload["stopped_reason"] == "budget"
        assert payload["stopped_early"] is False
        assert payload["n_steps"] == 10

    def test_rollout_without_a_condition_still_reports_budget(self, sim):
        """``stopped_reason`` is always present, so callers can branch on it."""
        payload = rollout_json(sim.run_policy(robot_name="alice", n_steps=5, fast_mode=True))
        assert payload["stopped_reason"] == "budget"

    def test_cancellation_reports_cancelled_not_predicate(self, sim):
        """A cooperative stop (e.g. ``stop_policy``) is a different outcome."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("alice"))

        def cancel_on_third_step(step: int, _obs: dict, _action: dict) -> None:
            if step >= 2:
                raise CooperativeStop

        result = PolicyRunner(sim).run(
            "alice",
            policy,
            n_steps=BUDGET,
            control_frequency=50,
            fast_mode=True,
            on_frame=cancel_on_third_step,
            stop_when=lambda _engine: False,
        )
        payload = rollout_json(result)
        assert result["status"] == "success"
        assert payload["stopped_early"] is True
        assert payload["stopped_reason"] == "cancelled"


class TestStopWhenRejection:
    """A condition the rollout cannot honor is a caller error, never a no-op."""

    def test_unknown_predicate_is_rejected_before_the_policy_runs(self, sim):
        policy = MagicMock()
        result = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            policy_object=policy,
            stop_when={"predicate": "cube_lifted", "body": "cube"},
        )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "stop_when" in text and "cube_lifted" in text
        # The valid set is listed, so the caller can self-correct.
        assert "body_above_z" in text
        # Nothing ran: rejection happens before the policy is driven.
        policy.get_actions.assert_not_called()

    def test_clause_naming_no_predicate_is_rejected(self, sim):
        """An empty clause would compile to a condition that never fires."""
        for empty in ({}, {"all": []}, {"any": []}):
            result = sim.run_policy(robot_name="alice", n_steps=5, stop_when=empty)
            assert result["status"] == "error", empty
            assert "names no predicate" in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", ["grasped", ["grasped"], 3])
    def test_non_mapping_clause_is_rejected(self, sim, bad):
        result = sim.run_policy(robot_name="alice", n_steps=5, stop_when=bad)
        assert result["status"] == "error"
        assert "stop_when" in result["content"][0]["text"]

    def test_single_call_mixed_with_a_group_is_rejected(self, sim):
        """``{predicate: ..., any: [...]}`` is ambiguous - half of it would be dropped."""
        result = sim.run_policy(
            robot_name="alice",
            n_steps=5,
            stop_when={
                "predicate": "body_below_z",
                "body": "cube",
                "z": CUBE_STOP_Z,
                "any": [{"predicate": "body_above_z", "body": "cube", "z": 1.0}],
            },
        )
        assert result["status"] == "error"
        assert "cannot also carry" in result["content"][0]["text"]

    def test_reward_term_is_rejected_as_a_stop_condition(self, sim):
        """A float-valued term read as a bool would stop the rollout instantly."""
        result = sim.run_policy(
            robot_name="alice",
            n_steps=5,
            stop_when={"predicate": "distance_neg", "body_a": "cube", "body_b": "cube"},
        )
        assert result["status"] == "error"
        assert "reward term" in result["content"][0]["text"]


class TestStopWhenComposesWithCapture:
    def test_video_holds_exactly_the_executed_steps(self, sim, tmp_path):
        """The frame that satisfied the condition is in the MP4."""
        path = tmp_path / "rollout.mp4"
        result = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            fast_mode=True,
            control_frequency=30.0,
            # One captured frame per control step (fps == control_frequency), so
            # the MP4 frame count IS the executed step count.
            video={"path": str(path), "fps": 30, "width": 160, "height": 120},
            stop_when={"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
        )
        payload = rollout_json(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["video_frames"] == payload["n_steps"]
        assert payload["video_path"] == str(path)
        assert path.exists() and path.stat().st_size > 0

    def test_recorded_dataset_holds_exactly_the_executed_steps(self, sim, tmp_path):
        """Round trip: a stopped rollout records the frames it ran, no more."""
        pytest.importorskip("lerobot")
        root = tmp_path / "ds"
        assert (
            sim.start_recording(repo_id="local/stop_when", task="drop", fps=30, root=str(root))["status"] == "success"
        )
        result = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            fast_mode=True,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
        )
        payload = rollout_json(result)
        assert sim.stop_recording()["status"] == "success"

        info_files = list(Path(root).rglob("meta/info.json"))
        assert info_files, f"no dataset written under {root}"
        info = json.loads(info_files[0].read_text())
        assert info["total_episodes"] == 1
        assert info["total_frames"] == payload["n_steps"]

    def test_multi_episode_run_applies_the_condition_per_episode(self, sim):
        """Each episode ends on the condition, so N episodes are N goal-reaching rollouts."""
        result = sim.run_policy(
            robot_name="alice",
            n_steps=BUDGET,
            n_episodes=2,
            fast_mode=True,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
        )
        assert result["status"] == "success"
        episodes = rollout_json(result)["episodes"]
        assert len(episodes) == 2
        assert [ep["stopped_reason"] for ep in episodes] == ["predicate", "predicate"]
        assert all(ep["n_steps"] < BUDGET for ep in episodes)


class TestStopWhenIsReachableFromAnAgent:
    def test_agent_tool_dispatch_forwards_the_clause(self, sim):
        """The router must forward ``stop_when``, not drop it as unknown."""
        result = sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "alice",
                "n_steps": BUDGET,
                "fast_mode": True,
                "stop_when": {"predicate": "body_below_z", "body": "cube", "z": CUBE_STOP_Z},
            },
        )
        assert result["status"] == "success"
        payload = rollout_json(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["n_steps"] < BUDGET

    def test_tool_spec_advertises_stop_when(self):
        """An LLM only forms calls from the schema it is handed."""
        spec_path = Path(__file__).resolve().parents[2] / "strands_robots/simulation/mujoco/tool_spec.json"
        props = json.loads(spec_path.read_text())["properties"]
        assert "stop_when" in props, "tool_spec.json must advertise run_policy's stop_when"
        assert props["stop_when"]["type"] == "object"

    def test_describe_documents_stop_when(self, sim):
        """``describe()`` is the discovery surface for the run_policy signature."""
        assert "stop_when" in sim.describe()["methods"]["run_policy"]
