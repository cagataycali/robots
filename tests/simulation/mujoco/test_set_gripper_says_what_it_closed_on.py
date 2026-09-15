"""``set_gripper(state="close")`` reports what the fingers closed on.

An agent running the README quickstart closed the gripper next to the cube,
lifted, and reported a successful pick - the reply "gripper commanded close"
reads the same whether the fingers met an object or air. The MuJoCo backend
now reads the contacts after the last tick: bodies outside the robot that
touch the finger subtree are named with their contact counts, and a close
that touched nothing says so and points at the fix.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def so100():
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    s.add_robot("so100", data_config="so100")
    s.add_object("red_cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.15, -0.15, 0.02], mass=0.1)
    yield s
    s.cleanup()


def _parts(result):
    assert result["status"] == "success", result
    return result["content"][0]["text"], result["content"][1]["json"]


def test_close_on_air_says_nothing_is_held(so100):
    text, payload = _parts(so100.set_gripper(robot_name="so100", state="close"))
    assert "Closed on nothing" in text
    assert "move_to the object first" in text
    assert payload["holding"] == []
    assert payload["finger_contacts"] == {}


def test_close_on_an_object_names_it(so100):
    import mujoco as mj

    so100.set_gripper(robot_name="so100", state="open", steps=20)
    model, data = so100._world._model, so100._world._data
    jaw = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "so100/Moving_Jaw")
    # A fixed post where the moving jaw is: the fingers close into it and
    # cannot push it away, so the reading does not depend on contact physics.
    so100.add_object(
        "post", shape="box", size=[0.03, 0.03, 0.03], position=[float(v) for v in data.xpos[jaw]], is_static=True
    )
    text, payload = _parts(so100.set_gripper(robot_name="so100", state="close", steps=30))
    assert "Closed on 'post'" in text
    assert "Closed on nothing" not in text
    assert payload["holding"] == ["post"]
    assert payload["finger_contacts"]["post"] >= 1


def test_open_does_not_claim_to_hold_anything(so100):
    text, payload = _parts(so100.set_gripper(robot_name="so100", state="open"))
    assert "Closed on" not in text
    assert "holding" not in payload


def test_envelope_without_contact_reading_is_unchanged():
    """Backends that do not pass ``held`` (Isaac) keep the previous reply."""
    result = MotionPrimitivesCore._set_gripper_result(
        "r", "close", 12, ["g"], {"g": 0.0}, {"g": "ctrlrange"}, {"j": 0.0}
    )
    assert result["content"][0]["text"] == "set_gripper: 'r' gripper commanded close (12 ticks, actuators ['g'])."
    assert "holding" not in result["content"][1]["json"]
