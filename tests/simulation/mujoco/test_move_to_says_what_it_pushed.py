"""``move_to`` says when the arm pushed an object off its target.

The README pick on ``so100``, scripted - open, ``move_to`` the cube's centre,
close - answered "reached" for the move and "Closed on nothing ... move_to the
object first" for the close. The straight servo descent had shoved the cube
2.6 cm along +Y, so the advice sent the caller back to the same target and the
same push. Pinned here: the move names the body it pushed and where it is now,
the close names where the nearest object went, and following that position
puts the cube between the fingers; a move that touches nothing adds nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

CUBE = [0.0, -0.2, 0.025]


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _text(result: dict) -> str:
    return " ".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _ok(result: dict, what: str) -> dict:
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {_text(result)}")
    return result


@pytest.fixture
def so100():
    sim = MuJoCoSimEngine(tool_name="pushed_so100", mesh=False)
    _ok(sim.create_world(), "create_world")
    _ok(sim.add_robot(name="so100", data_config="so100"), "add_robot")
    _ok(sim.add_object("red_cube", shape="box", size=[0.05] * 3, position=CUBE, color=[1, 0, 0]), "add_object")
    try:
        yield sim
    finally:
        sim.cleanup()


def _cube(sim) -> np.ndarray:
    model, data = sim._world._model, sim._world._data
    return np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")])


def test_the_readme_pick_names_the_push_and_where_the_cube_went(so100):
    _ok(so100.set_gripper(robot_name="so100", state="open"), "open")
    move = _ok(so100.move_to(robot_name="so100", position=CUBE), "move_to")

    (record,) = _json(move)["pushed"]
    assert record["body"] == "red_cube"
    assert record["moved_m"] >= 0.005
    assert np.allclose(record["position"], _cube(so100))
    assert "The arm pushed 'red_cube'" in _text(move)

    close = _ok(so100.set_gripper(robot_name="so100", state="close"), "close")
    assert _json(close)["holding"] == [], "premise: the pushed cube is out of the fingers"
    nearest = _json(close)["nearest_object"]
    assert nearest["body"] == "red_cube" and np.allclose(nearest["position"], _cube(so100))
    assert "move_to the object first" not in _text(close)

    _ok(so100.set_gripper(robot_name="so100", state="open"), "reopen")
    _ok(so100.move_to(robot_name="so100", position=_cube(so100).tolist()), "retarget")
    regrip = _ok(so100.set_gripper(robot_name="so100", state="close"), "regrip")
    assert "red_cube" in _json(regrip)["holding"], "following the reported position reaches the cube"


def test_a_move_that_touches_nothing_reports_no_push(so100):
    move = _ok(so100.move_to(robot_name="so100", position=[0.1, -0.15, 0.15], tol=0.02), "move_to")
    assert "pushed" not in _json(move)
    assert "pushed" not in _text(move)
