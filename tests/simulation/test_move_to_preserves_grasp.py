"""Regression tests: move_to actually carries a grasped object.

``move_to`` documents a grasp-preservation contract - "``set_gripper("close")``
-> ``move_to(...)`` carries the held object rather than releasing it". It did not
hold, for two independent reasons, and the failure was silent: every primitive
returned ``status="success"`` while the object stayed on the floor.

1. **The gripper channel was omitted for a tendon drive.** The hold-the-gripper
   comprehension iterated ``_joint_actuator_map``, which by construction only
   contains JOINT/JOINTINPARENT transmissions. The Panda's gripper - and every
   split gripper like it - is a TENDON actuator, so it was left out of
   ``ctrl_targets`` entirely: exactly the case the surrounding comment says must
   not happen ("an unwritten channel would let a stale ctrl from another path
   drive it"). Fixed by iterating ``grip_acts``, the authoritative set.

2. **The hold was seeded from ``data.qpos``, not ``data.ctrl``.** Re-commanding a
   position servo to where the grasped object is holding it OPEN means "stop
   squeezing", so the fingers crept shut through the object.

3. **The arm command was a step, not a ramp.** A position servo handed a far
   set-point applies full gain at once. Measured on the Panda lifting a 4 cm
   cube: 5.9 rad/s peak joint velocity within 5 control ticks, and the jerk tore
   the cube out of the gripper. The grasp itself was never the problem - 3.28x
   static friction margin, and stationary it held 600 steps with zero drift.

Before / after, same scene (Panda, 50 g 4 cm cube, top-down grasp):

    step command:   cube_z 0.0099 -> 0.0099   fingers 0.00997 -> 0.00163  DROPPED
    ramped command: cube_z 0.0099 -> 0.2348   fingers 0.00997 -> 0.00998  LIFTED

Verified by render: the cube sits in the fingertips at z=0.2492, clear of the floor.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.mujoco.motion_primitives import _RAMP_RAD_PER_TICK  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Top-down grasp orientation: the hand's -Z (its approach axis) points at the
# floor, which is a 180 deg rotation about X from identity.
_DOWN = [0.0, 1.0, 0.0, 0.0]

# The Panda has no TCP site, so the discovered EE frame is the ``hand`` BODY,
# which sits ~0.104 m above the fingertips. Grasping a cube resting at z=0.02
# therefore means commanding the hand to 0.124.
_GRASP_Z = 0.124


@pytest.fixture
def sim():
    s = Simulation(tool_name="move_to_preserves_grasp", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert (
        s.add_object(name="cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.45, 0.0, 0.02], mass=0.05)["status"]
        == "success"
    )
    s.step(n_steps=300)
    yield s
    s.destroy()


def _cube_z(sim) -> float:
    body = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return float(sim.mj_data.xpos[body][2])


def _finger_qpos(sim) -> list[float]:
    model, data = sim.mj_model, sim.mj_data
    return [
        float(data.qpos[int(model.jnt_qposadr[j])])
        for j in range(model.njnt)
        if "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
    ]


def _grasp_the_cube(sim) -> None:
    """Open, descend, close - leaving the cube held between the fingertips."""
    assert sim.set_gripper(robot_name="a", state="open", steps=60)["status"] == "success"
    assert (
        sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.10], orientation=_DOWN, tol=0.015, max_steps=900)[
            "status"
        ]
        == "success"
    )
    assert (
        sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z], orientation=_DOWN, tol=0.015, max_steps=900)[
            "status"
        ]
        == "success"
    )
    assert sim.set_gripper(robot_name="a", state="close", steps=200)["status"] == "success"
    # Premise: the fingers closed onto the cube's half-width (0.02 m), not shut.
    assert all(0.005 < q < 0.015 for q in _finger_qpos(sim)), _finger_qpos(sim)


def test_a_grasped_cube_is_actually_lifted(sim) -> None:
    """The core defect: this used to leave the cube on the floor, reporting success."""
    _grasp_the_cube(sim)
    assert _cube_z(sim) < 0.05, "premise: the cube starts on the floor"

    result = sim.move_to(
        robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900
    )
    assert result["status"] == "success", result["content"][0]["text"]
    assert _cube_z(sim) > 0.15, f"the cube was left behind (z={_cube_z(sim):.4f})"


def test_the_lift_survives_settling(sim) -> None:
    """Still held after the arm stops - not merely flicked upward in passing."""
    _grasp_the_cube(sim)
    sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900)
    sim.step(n_steps=500)
    assert _cube_z(sim) > 0.15, f"the cube slipped while holding (z={_cube_z(sim):.4f})"


def test_the_object_can_be_carried_laterally(sim) -> None:
    """Transport, not just a vertical lift - the documented use case."""
    _grasp_the_cube(sim)
    sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900)
    result = sim.move_to(
        robot_name="a", position=[0.30, 0.20, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.03, max_steps=900
    )
    assert result["status"] == "success", result["content"][0]["text"]
    sim.step(n_steps=400)
    assert _cube_z(sim) > 0.15, f"the cube was dropped in transit (z={_cube_z(sim):.4f})"


def test_the_fingers_do_not_creep_shut_through_the_object(sim) -> None:
    """Bug 2's signature: the jaw squeezing past the object it is holding.

    Before the fix the fingers went 0.00997 -> 0.00163 m within 25 control ticks
    of move_to, passing straight through a cube whose half-width is 0.02 m.
    """
    _grasp_the_cube(sim)
    before = _finger_qpos(sim)
    sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900)
    after = _finger_qpos(sim)
    for q0, q1 in zip(before, after, strict=True):
        assert abs(q1 - q0) < 2e-3, f"finger travelled {q0:.5f} -> {q1:.5f} during move_to"


def test_a_tendon_gripper_is_commanded_at_all(sim) -> None:
    """Bug 1 directly: the Panda's gripper is a TENDON drive, so the old
    joint-transmission-only comprehension omitted it from ctrl_targets."""
    model = sim.mj_model
    robot = sim._world.robots["a"]
    grip_acts, _, err = sim._resolve_gripper_actuators(model, robot)
    assert err is None and grip_acts
    joint_driven = set(sim._joint_actuator_map(model, robot).values())
    assert not (grip_acts & joint_driven), "premise changed: the Panda gripper is no longer tendon-driven"

    captured: list[dict[int, float]] = []
    original = type(sim)._primitive_tick

    def capture(self, m, d, ctrl):  # noqa: ANN001, ANN202
        captured.append(dict(ctrl))
        return original(self, m, d, ctrl)

    type(sim)._primitive_tick = capture
    try:
        sim.move_to(robot_name="a", position=[0.45, 0.0, 0.35], orientation=_DOWN, tol=0.02, max_steps=30)
    finally:
        type(sim)._primitive_tick = original

    assert captured, "move_to never ticked"
    for act_id in grip_acts:
        assert act_id in captured[0], f"gripper actuator {act_id} was not commanded"


def test_the_gripper_hold_uses_the_command_not_the_measurement(sim) -> None:
    """Bug 2's mechanism, pinned on the source.

    Holding at ``data.qpos`` re-commands the servo to where the OBJECT is forcing
    the jaw, which reads as "stop squeezing"; the command (``data.ctrl``) is what
    set_gripper asked for and is a true no-op on the grasp.
    """
    import inspect

    source = inspect.getsource(Simulation.move_to)
    assert "float(data.ctrl[act_id]) for act_id in grip_acts" in source


def test_the_ramp_bounds_the_commanded_velocity(sim) -> None:
    """Bug 3: peak joint velocity must stay far below the un-ramped 5.9 rad/s."""
    _grasp_the_cube(sim)
    peak = 0.0
    original = type(sim)._primitive_tick

    def watch(self, m, d, ctrl):  # noqa: ANN001, ANN202
        nonlocal peak
        result = original(self, m, d, ctrl)
        peak = max(peak, float(np.abs(d.qvel[:7]).max()))
        return result

    type(sim)._primitive_tick = watch
    try:
        sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900)
    finally:
        type(sim)._primitive_tick = original
    assert peak < 2.0, f"peak joint velocity {peak:.2f} rad/s - the ramp is not bounding the command"


def test_a_short_move_is_not_slowed_to_transport_pace(sim) -> None:
    """The ramp is sized by joint travel, so a small correction stays quick."""
    assert (
        sim.move_to(robot_name="a", position=[0.45, 0.0, 0.35], orientation=_DOWN, tol=0.02, max_steps=900)["status"]
        == "success"
    )
    result = sim.move_to(robot_name="a", position=[0.45, 0.0, 0.36], orientation=_DOWN, tol=0.02, max_steps=900)
    assert result["status"] == "success"
    payload = next(b["json"] for b in result["content"] if "json" in b)
    assert payload["steps"] < 200, f"a 1 cm correction took {payload['steps']} ticks"


def test_the_ramp_cannot_consume_the_whole_step_budget() -> None:
    """A tiny max_steps must still leave ticks for free servoing."""
    import inspect

    source = inspect.getsource(Simulation.move_to)
    assert "max(1, max_steps // 2)" in source
    assert _RAMP_RAD_PER_TICK > 0


def test_rotate_wrist_also_commands_a_tendon_gripper(sim) -> None:
    """The same omission lived in rotate_wrist.

    ``rotate_wrist`` builds its hold set by iterating ``_joint_actuator_map``, so
    a tendon-driven gripper was never written there either. It happens not to
    drop the object (every arm joint is held, so there is no jerk), but leaving a
    channel unwritten is the documented hazard: "an unwritten channel would let a
    stale ctrl from another path drive it".
    """
    _grasp_the_cube(sim)
    model = sim.mj_model
    grip_acts, _, err = sim._resolve_gripper_actuators(model, sim._world.robots["a"])
    assert err is None and grip_acts

    captured: list[dict[int, float]] = []
    original = type(sim)._primitive_tick

    def capture(self, m, d, ctrl):  # noqa: ANN001, ANN202
        captured.append(dict(ctrl))
        return original(self, m, d, ctrl)

    type(sim)._primitive_tick = capture
    try:
        sim.rotate_wrist(robot_name="a", target_yaw=0.4, tol=0.03, max_steps=60)
    finally:
        type(sim)._primitive_tick = original

    assert captured, "rotate_wrist never ticked"
    for act_id in grip_acts:
        assert act_id in captured[0], f"gripper actuator {act_id} was not commanded by rotate_wrist"


def test_rotate_wrist_keeps_holding_the_object(sim) -> None:
    """Behavioural counterpart: the grasp survives a wrist rotation."""
    _grasp_the_cube(sim)
    sim.move_to(robot_name="a", position=[0.45, 0.0, _GRASP_Z + 0.25], orientation=_DOWN, tol=0.025, max_steps=900)
    assert _cube_z(sim) > 0.15, "premise: the cube must be lifted first"
    result = sim.rotate_wrist(robot_name="a", target_yaw=0.8, tol=0.03, max_steps=600)
    assert result["status"] == "success", result["content"][0]["text"]
    sim.step(n_steps=300)
    assert _cube_z(sim) > 0.15, f"the cube was dropped by rotate_wrist (z={_cube_z(sim):.4f})"
