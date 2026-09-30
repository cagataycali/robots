"""``move_to`` says when what the fingers held did not come along.

On the shipped SO-100 / SO-101 a two-finger friction pinch does not lift a cube
in MuJoCo (#2167, #2145; ``examples/18_so101_pick_and_lift.py`` carries it with
``attach_bodies(mode="weld")``). Nothing told an agent that: ``set_gripper``
answered "Closed on 'cube' (4 contacts)" and the lift ``move_to`` answered
"reached ... success" while the cube stayed on the table (1.6 mm on so101, 0.0 mm
on so100). A Bedrock agent asked to pick the cube up spent 102 tool calls
re-grasping and editing geom friction and never found ``attach_bodies``.

Pinned here: the lift reports the body it left behind, with the weld parent to
use; following that advice lifts the cube; and a move that holds nothing, or
travels too little to tell, adds nothing.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _text(result: dict) -> str:
    return " ".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _ok(result: dict, what: str) -> dict:
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {_text(result)}")
    return result


@pytest.fixture(params=["so101", "so100"])
def arm_at_cube(request):
    robot = request.param
    sim = MuJoCoSimEngine(tool_name=f"grasp_{robot}", mesh=False)
    _ok(sim.create_world(), "create_world")
    _ok(sim.add_robot(name=robot, data_config=robot), "add_robot")
    x, y = _json(sim.get_robot_state(robot_name=robot))["end_effector"]["position"][:2]
    _ok(sim.add_object("cube", shape="box", size=[0.025] * 3, position=[x, y, 0.0125], mass=0.05), "add_object")
    sim.step(200)
    z0 = _cube_z(sim)
    _ok(sim.set_gripper(robot_name=robot, state="open"), "open")
    _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.10], tol=0.02), "hover")
    _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.005], tol=0.02), "descend")
    closed = _ok(sim.set_gripper(robot_name=robot, state="close"), "close")
    assert "cube" in _json(closed)["holding"], "premise: the fingers close on the cube"
    try:
        yield sim, robot, (x, y, z0)
    finally:
        sim.cleanup()


def _cube_z(sim) -> float:
    model, data = sim._world._model, sim._world._data
    return float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")][2])


def test_the_lift_reports_the_cube_it_left_behind(arm_at_cube):
    sim, robot, (x, y, z0) = arm_at_cube

    lift = _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.12], tol=0.02), "lift")

    assert _cube_z(sim) - z0 < 0.01, "premise: the friction pinch does not lift on this model"
    (record,) = _json(lift)["left_behind"]
    assert record["body"] == "cube"
    assert record["ee_moved_m"] > 0.05 and record["body_moved_m"] < 0.5 * record["ee_moved_m"]
    assert "'cube' was in the fingers when the move started and was left behind" in _text(lift)
    assert f"attach_bodies(parent='{record['weld_parent']}', child='cube', mode='weld')" in _text(lift)


def test_following_the_advice_lifts_the_cube(arm_at_cube):
    sim, robot, (x, y, z0) = arm_at_cube
    record = _json(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.12], tol=0.02))["left_behind"][0]

    _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.005], tol=0.02), "back down")
    _ok(sim.set_gripper(robot_name=robot, state="close"), "close")
    _ok(sim.attach_bodies(parent=record["weld_parent"], child="cube", mode="weld"), "attach")
    lift = _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.12], tol=0.02), "lift")

    assert _cube_z(sim) - z0 > 0.05
    assert "left_behind" not in _json(lift)


def test_a_move_holding_nothing_or_barely_moving_adds_nothing(arm_at_cube):
    sim, robot, (x, y, z0) = arm_at_cube
    nudge = _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.01], tol=0.02), "nudge")
    assert "left_behind" not in _json(nudge), "under 2 cm of travel proves nothing"

    _ok(sim.set_gripper(robot_name=robot, state="open"), "open")
    _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.10], tol=0.02), "away")
    empty = _ok(sim.move_to(robot_name=robot, position=[x, y, z0 + 0.03], tol=0.02), "empty move")
    assert "left_behind" not in _json(empty)
