"""A rollout through an action controller records the joint command it applied.

With a task-space controller installed (``IsaacDeltaEEFController``), a policy
emits ``{x, y, z, roll, pitch, yaw, gripper}`` while the dataset's action
columns are the robot's joints. The recording hook stored the policy's dict, so
every joint column was absent and the recorder refused every frame: a π0.5-LIBERO
rollout recorded 0 frames while the arm moved. The frame now stores the joint
position target standing on EVERY joint after the command - the controller's
output merged over the targets before it, seeded from the measured positions.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402

JOINTS = ["j0", "j1", "j2"]


class _Articulation:
    def __init__(self) -> None:
        self.q = np.array([0.1, 0.2, 0.3])
        self.applied: list[tuple[list[int], list[float]]] = []

    def get_joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def apply_action(self, action: Any) -> None:
        self.applied.append((list(action.joint_indices), list(action.joint_positions)))


class _Controller:
    """Names only the joints a step moves, as the delta-EEF controller does."""

    def __init__(self) -> None:
        self.out: dict[str, float] = {"j1": 0.5}

    def compute_joint_targets(self, action: Any) -> dict[str, float]:
        return dict(self.out)


@pytest.fixture
def engine(monkeypatch) -> Any:
    import sys

    fake = types.ModuleType("strands_robots.simulation.isaac._deprecated_api")
    fake.ArticulationAction = lambda joint_positions, joint_indices: types.SimpleNamespace(  # type: ignore[attr-defined]
        joint_positions=joint_positions, joint_indices=joint_indices
    )
    monkeypatch.setitem(sys.modules, "strands_robots.simulation.isaac._deprecated_api", fake)
    eng = isaac_engine()
    eng._world = types.SimpleNamespace(step=lambda render=False: None)
    eng._world_created = True
    eng._STEPS_PER_BATCH = IsaacSimulation._STEPS_PER_BATCH
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=list(JOINTS))
    robot.articulation = _Articulation()
    eng._robots["arm"] = robot
    return eng


def test_without_a_controller_the_policy_action_is_recorded_as_is(engine) -> None:
    action = {"j0": 1.0}
    assert engine._recorded_action("arm", action) is action


def test_the_first_command_records_every_joint_standing_target(engine) -> None:
    engine._action_controllers["arm"] = _Controller()

    assert engine.send_action({"x": 0.01, "gripper": 1.0}, robot_name="arm")["status"] == "success"

    # j1 is what the controller commanded; j0 and j2 hold where they stood.
    assert engine._recorded_action("arm", {"x": 0.01}) == pytest.approx({"j0": 0.1, "j1": 0.5, "j2": 0.3})


def test_later_commands_merge_over_the_targets_before_them(engine) -> None:
    controller = _Controller()
    engine._action_controllers["arm"] = controller
    engine.send_action({"x": 0.01}, robot_name="arm")
    engine._robots["arm"].articulation.q = np.array([9.0, 9.0, 9.0])  # measured drifts; targets do not

    controller.out = {"j2": -0.4}
    engine.send_action({"z": -0.01}, robot_name="arm")

    assert engine._recorded_action("arm", {"z": -0.01}) == pytest.approx({"j0": 0.1, "j1": 0.5, "j2": -0.4})


def test_a_step_the_controller_names_no_joint_for_still_records_the_standing_targets(engine) -> None:
    controller = _Controller()
    engine._action_controllers["arm"] = controller
    engine.send_action({"x": 0.01}, robot_name="arm")
    controller.out = {}  # an all-zero delta: hold

    engine.send_action({"x": 0.0}, robot_name="arm")

    assert engine._recorded_action("arm", {"x": 0.0}) == pytest.approx({"j0": 0.1, "j1": 0.5, "j2": 0.3})


def test_the_recording_hook_hands_the_recorder_joint_columns(engine) -> None:
    engine._action_controllers["arm"] = _Controller()
    frames: list[dict[str, Any]] = []
    recorder = types.SimpleNamespace(add_frame=lambda **kw: frames.append(kw))
    state = engine._recording_state()
    state.update(recording=True, dataset_recorder=recorder, recording_cameras=[], trajectory=[])
    engine.robot_action_keys = lambda name: list(JOINTS)  # type: ignore[method-assign]
    hook = engine._make_recording_on_frame("arm", "probe")

    engine.send_action({"x": 0.01}, robot_name="arm")
    hook(0, {"j0": 0.1, "j1": 0.2, "j2": 0.3}, {"x": 0.01, "gripper": 1.0})

    assert frames[0]["action"] == pytest.approx({"j0": 0.1, "j1": 0.5, "j2": 0.3})
    # The trajectory keeps the policy's own task-space action.
    assert state["trajectory"][0].action == {"x": 0.01, "gripper": 1.0}


class _BlindArticulation(_Articulation):
    """A read that fails, as it does while the physics view is being rebuilt."""

    def get_joint_positions(self) -> np.ndarray:
        raise RuntimeError("physics view not ready")


class _MuteArticulation(_Articulation):
    def get_joint_positions(self) -> None:  # type: ignore[override]
        return None


@pytest.mark.parametrize("articulation", [_BlindArticulation, _MuteArticulation], ids=["raises", "none"])
def test_a_failed_measured_read_seeds_only_the_joints_the_command_names(engine, articulation, caplog) -> None:
    """No zero stands in for a position nobody measured (Key Conventions #6).

    The standing targets are seeded from the measured positions. When that read
    fails, the unnamed joints have NO target: the recorder refuses the frame
    for the columns without a value, instead of a fabricated home pose of
    zeros flowing into every frame of the episode.
    """
    engine._robots["arm"].articulation = articulation()
    engine._action_controllers["arm"] = _Controller()

    with caplog.at_level("WARNING", logger="strands_robots.simulation.isaac.simulation"):
        assert engine.send_action({"x": 0.01}, robot_name="arm")["status"] == "success"

    recorded = engine._recorded_action("arm", {"x": 0.01})
    assert dict(recorded) == pytest.approx({"j1": 0.5}), recorded
    assert "j0" not in recorded and "j2" not in recorded
    assert any("measured positions" in r.message and "arm" in r.message for r in caplog.records), caplog.text


def test_a_failed_seed_is_retried_on_the_next_command(engine) -> None:
    """Once the read works, the still-unnamed joints get their real standing target."""
    engine._robots["arm"].articulation = _BlindArticulation()
    controller = _Controller()
    engine._action_controllers["arm"] = controller
    engine.send_action({"x": 0.01}, robot_name="arm")
    assert dict(engine._recorded_action("arm", {"x": 0.01})) == pytest.approx({"j1": 0.5})

    engine._robots["arm"].articulation = _Articulation()
    controller.out = {"j2": -0.4}
    engine.send_action({"z": -0.01}, robot_name="arm")

    assert engine._recorded_action("arm", {"z": -0.01}) == pytest.approx({"j0": 0.1, "j1": 0.5, "j2": -0.4})
