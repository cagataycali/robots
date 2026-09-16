"""A joint write says what unit its numbers are in, and names a degree reading.

``set_joint_positions`` takes a MuJoCo joint coordinate: radians on a hinge,
metres on a slide. Nothing said so. The published ``positions`` description read
"Joint name -> position mapping for set_joint_positions", the range refusal read
``shoulder_pan=-96.2 outside [-1.92, 1.92]``, and neither named a unit -- so a
caller mirroring a real arm onto its sim twin had to already know the convention.
The driver on the other side reports the other unit (``drivers/feetech`` reads an
SO-arm in degrees), which makes the mismatch the ordinary case rather than an
exotic one, and the cheapest reading of a bare ``outside [-1.92, 1.92]`` is
"clamp to the bound" -- a pose 110 degrees from the one intended.

The unit is a property of the joint's TYPE, so it is derived from the model
(:func:`~strands_robots.simulation.mujoco.scene_ops.joint_position_unit`) rather
than asserted as prose: a blanket "radians" would be wrong on every slide joint,
of which the shipped grippers and telescoping arms have plenty.

These cells pin the two surfaces the caller reads (the published description and
the refusal), that the degree sentence states a fact rather than a guess -- it
appears only when converting the value really does land inside the range -- and
the unit rule itself over all four joint types.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import joint_position_unit  # noqa: E402

# A pose read off a real SO-101 in the unit its driver reports: degrees. Every
# entry is outside its joint's range when read as radians, and inside it once
# converted, which is exactly the case the refusal has to be able to name.
REAL_ARM_DEGREES = {
    "shoulder_pan": -96.2,
    "shoulder_lift": -88.0,
    "elbow_flex": 85.0,
    "wrist_flex": 40.0,
    "wrist_roll": -12.0,
    "gripper": 20.0,
}

# One slide joint, limited in metres, so "radians" would be the wrong word for it.
SLIDE_XML = """
<mujoco model="lift">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.02" mass="1"/>
      <body name="carriage" pos="0 0 0.05">
        <joint name="rise" type="slide" axis="0 0 1" range="0 0.04" limited="true" damping="1"/>
        <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="rise_act" joint="rise" kp="30" dampratio="1"/>
  </actuator>
</mujoco>
"""

# One joint of every MuJoCo type, so the unit rule is read off the type rather
# than off the one arm the other cells use.
EVERY_JOINT_TYPE_XML = """
<mujoco model="types">
  <compiler angle="radian"/>
  <worldbody>
    <body name="swing" pos="0 0 0.5">
      <joint name="a_hinge" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="track" pos="0.2 0 0.5">
      <joint name="a_slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="socket" pos="0.4 0 0.5">
      <joint name="a_ball" type="ball"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="loose" pos="0.6 0 0.5">
      <joint name="a_free" type="free"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _arm():
    """A world holding the registry's so101, whose joints are all hinges."""
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco", tool_name="unit_sim", mesh=False)
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot(name="so101")["status"] == "success"
    return sim


def _inline(tmp_path: Path, name: str, xml: str):
    """A world holding exactly one robot compiled from inline MJCF."""
    from strands_robots.simulation import create_simulation

    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    sim = create_simulation("mujoco", tool_name="unit_sim", mesh=False)
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot(name=name, urdf_path=str(path))["status"] == "success"
    return sim


def test_the_range_refusal_names_the_unit_of_the_bounds_it_prints() -> None:
    sim = _arm()
    res = sim.set_joint_positions(positions={"shoulder_pan": -96.2}, robot_name="so101")
    text = res["content"][0]["text"]
    assert res["status"] == "error", text
    # The unit sits on the bounds, not loose in the sentence: the numbers are
    # what the caller compares its own value against.
    assert "outside [-1.92, 1.92] rad" in text, text


def test_a_degree_reading_that_converts_inside_the_range_is_named_as_one() -> None:
    sim = _arm()
    res = sim.set_joint_positions(positions=dict(REAL_ARM_DEGREES), robot_name="so101")
    text = res["content"][0]["text"]
    assert res["status"] == "error", text
    for label, degrees in REAL_ARM_DEGREES.items():
        # Every joint of the pose is named, with the conversion that lands it
        # inside the range - the whole mirror is refused, not just the first.
        assert f"{label}={degrees:.4g} outside" in text, (label, text)
        assert f"{degrees:.4g} deg = {math.radians(degrees):.4g} rad" in text, (label, text)
    assert "radians, not degrees" in text, text
    # Nothing was written: the refusal is still all-or-nothing.
    assert list(sim._world._data.qpos) == pytest.approx([0.0] * 6)


def test_a_value_no_conversion_rescues_is_not_told_it_is_degrees() -> None:
    sim = _arm()
    # 1e6 rad is out of range, and so is 1e6 deg - so there is no degree reading
    # to name, and claiming one would be a guess about the caller's intent.
    res = sim.set_joint_positions(positions={"shoulder_pan": 1e6}, robot_name="so101")
    text = res["content"][0]["text"]
    assert res["status"] == "error", text
    assert "outside [-1.92, 1.92] rad" in text, text
    assert "deg" not in text, text


def test_a_slide_joint_range_refusal_is_in_metres_not_radians(tmp_path: Path) -> None:
    sim = _inline(tmp_path, "lift", SLIDE_XML)
    res = sim.set_joint_positions(positions={"rise": 5.0}, robot_name="lift")
    text = res["content"][0]["text"]
    assert res["status"] == "error", text
    assert "outside [0, 0.04] m" in text, text
    # A slide joint has no degree reading, so neither word belongs here.
    assert "rad" not in text, text
    assert "deg" not in text, text


def test_the_published_position_and_velocity_descriptions_name_the_unit() -> None:
    sim = _arm()
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    positions = props["positions"]["description"]
    velocities = props["velocities"]["description"]
    # An agent reads the schema before it reads a docstring, so both units and
    # the unit NOT to use have to be there.
    assert "radians" in positions and "metres" in positions and "degrees" in positions, positions
    assert "rad/s" in velocities and "m/s" in velocities, velocities


@pytest.mark.parametrize(
    ("joint", "unit"),
    [("a_hinge", "rad"), ("a_slide", "m"), ("a_ball", "rad"), ("a_free", "")],
)
def test_the_unit_is_read_off_the_joint_type(joint: str, unit: str) -> None:
    model = mujoco.MjModel.from_xml_string(EVERY_JOINT_TYPE_XML)
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    assert jnt_id >= 0
    assert joint_position_unit(model, jnt_id, mujoco) == unit
