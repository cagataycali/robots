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


_TENDON_GRIPPER_ONE_BODY = """
<mujoco model="one_body_tendon">
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.05"/>
      <site name="a" pos="0.02 0 0"/>
      <site name="b" pos="-0.02 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="gripper_cable" limited="false">
      <site site="a"/>
      <site site="b"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="gripper" tendon="gripper_cable" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""


def _pad_tip(model, data, body, hand, mj):
    """Centre of a finger's tip pad: its collision geom furthest from the hand."""
    pads = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and int(model.geom_group[g]) == 2]
    return max((data.geom_xpos[g] for g in pads), key=lambda p: float(sum((p - data.xpos[hand]) ** 2)))


def test_a_tendon_gripper_names_what_it_closed_on():
    """A Franka's two jaws share one tendon drive, so its fingers are no joint's body.

    A tendon transmission has no joint to read, so resolving the fingers from
    the drive's joint located none of them - and fingers that were never
    located touch nothing. The reply said "Closed on nothing" with the cube
    between the pads, the very claim this text exists to refute.

    The exact ``holding`` also pins the other half: the robot is not something
    it closed on. In this pose an arm link rests against the gripper, and it
    was offered as a grasp because the check excluded the robot's bodies by
    ``SimRobot.body_id``, which no MuJoCo robot sets (it is -1), so the
    exclusion was empty for every one of them.
    """
    import mujoco as mj

    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    s.add_robot("arm", data_config="panda")
    try:
        s.set_gripper(robot_name="arm", state="open", steps=30)
        model, data = s._world._model, s._world._data
        hand = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/hand"))
        tips = [
            _pad_tip(model, data, int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"arm/{side}_finger")), hand, mj)
            for side in ("left", "right")
        ]
        between = [float((tips[0][i] + tips[1][i]) / 2) for i in range(3)]
        # Static, so the reading does not depend on the grasp holding the cube.
        s.add_object("red_cube", shape="box", size=[0.03, 0.03, 0.03], position=between, is_static=True)

        text, payload = _parts(s.set_gripper(robot_name="arm", state="close", steps=60))
        assert "Closed on 'red_cube'" in text
        assert "Closed on nothing" not in text
        assert payload["holding"] == ["red_cube"]
        assert payload["finger_contacts"]["red_cube"] >= 1
    finally:
        s.cleanup()


def test_a_drive_that_reaches_no_body_says_nothing_about_holding(tmp_path):
    """Unlocatable fingers report nothing at all, never "nothing is held".

    A spatial tendon whose whole route sits on one body drives no joint, so no
    finger body resolves. An empty contact reading would then be a claim the
    backend never established, so this close replies exactly as it did before
    contacts were read.
    """
    path = tmp_path / "one_body_tendon.xml"
    path.write_text(_TENDON_GRIPPER_ONE_BODY)
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    s.add_robot("cable", urdf_path=str(path))
    try:
        text, payload = _parts(s.set_gripper(robot_name="cable", state="close", steps=5))
        assert "Closed on" not in text
        assert "holding" not in payload
        assert "finger_contacts" not in payload
    finally:
        s.cleanup()


_FINGER_ONTO_THE_FLOOR = """
<mujoco model="floor_toucher">
  <worldbody>
    <body name="mount" pos="0.3 0 0.06">
      <geom type="box" size="0.02 0.02 0.02"/>
      <body name="jaw">
        <joint name="jaw_slide" type="slide" axis="0 0 1" range="-0.06 0"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="gripper" joint="jaw_slide" ctrlrange="-0.06 0"/>
  </actuator>
</mujoco>
"""


def test_the_ground_is_not_something_the_gripper_closed_on(tmp_path):
    """Closing a gripper down onto the table is not a grasp of the world.

    The ground plane belongs to the world body, which every scene has and no
    lift can carry, so touching it leaves the reading empty rather than
    offering "world" as the thing held.
    """
    path = tmp_path / "floor_toucher.xml"
    path.write_text(_FINGER_ONTO_THE_FLOOR)
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    s.add_robot("toucher", urdf_path=str(path))
    try:
        text, payload = _parts(s.set_gripper(robot_name="toucher", state="close", steps=60))
        assert "Closed on nothing" in text
        assert "world" not in text
        assert payload["holding"] == []
    finally:
        s.cleanup()
