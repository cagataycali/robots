"""A cable routed over sites drives the joints it spans, and is read that way.

MJCF spells a tendon two ways. A ``<fixed>`` tendon lists joint coordinates, so
its wrap entries name joints outright. A ``<spatial>`` tendon lists a route of
*sites*, so its wrap entries name sites - and a site id names no joint at all.
``tendon_joint_ids`` kept only the joint entries, so every spatial tendon read as
driving nothing whatever it was wired to.

That is the "reads an actuated joint as free" failure
:func:`~strands_robots.simulation.mujoco.scene_ops.actuator_driven_joint_ids`
names in its own docstring, and it reached the surfaces a caller drives the robot
through. Measured on the shipped ``aero_hand`` asset (Tetheria Aero Hand Open,
16 joints, 7 actuators, 6 of them cables), loaded through ``load_scene``:

* those 6 actuators reported no driven joint, while MuJoCo's own
  ``qfrc_actuator`` moves 15 of the hand's 16 joints when their ``ctrl`` moves;
* ``send_action({"<finger joint>": x})`` therefore found no driving actuator for
  the joint and **dropped the value**, the silent-gripper-drop class issue #318
  was filed to fix, leaving one commandable joint of sixteen;
* ``actuate_robot`` saw nothing driven and would add a position servo per joint
  on top of the cable already pulling it, so the two fight - the double-actuation
  its own refusal exists to prevent;
* ``joint_drive_map`` placed those joints in neither the servo nor the
  other-drive map, so a pose write treats a cable-driven finger as free.

Every model here is inline MJCF loaded through ``add_robot(urdf_path=...)``, so
none of it downloads an asset, and the reading is graded against MuJoCo rather
than against the rule under test: ``_physically_driven`` moves one actuator's
``ctrl`` and reports the joints whose generalized force responds. Across the 554
MJCF files shipped in the asset cache that oracle agrees with the reading for all
75 tendon-transmission actuators, where before it agreed for 57 of 75 - the 18 it
disagreed with being every spatial drive in the corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    actuator_driven_joint_ids,
    actuator_joint_id,
    actuator_target_body_ids,
    joint_drive_map,
    tendon_joint_ids,
)
from tests.simulation.mujoco.test_actuate_robot_sees_a_tendon_drive import (  # noqa: E402
    SITE_DRIVE_XML,
    _lone,
)
from tests.simulation.mujoco.test_actuator_ownership_by_transmission import GRIPPER_XML  # noqa: E402

# A two-link finger closed by one cable anchored on the palm and routed over a
# site on each link: the tendon-driven-hand idiom, and the shape all 18 spatial
# drives in the shipped asset cache take.
FLEXOR_XML = """
<mujoco model="flexor">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="palm" pos="0 0 0.3">
      <geom type="box" size="0.05 0.02 0.02"/>
      <site name="anchor" pos="-0.04 0 0.02"/>
      <body name="prox" pos="0.05 0 0">
        <joint name="mcp" type="hinge" axis="0 1 0" range="-1.6 1.6" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.012"/>
        <site name="mid" pos="0.04 0 0.015"/>
        <body name="dist" pos="0.08 0 0">
          <joint name="pip" type="hinge" axis="0 1 0" range="-1.6 1.6" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.01"/>
          <site name="tip" pos="0.03 0 0.012"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="flexor" width="0.002">
      <site site="anchor"/>
      <site site="mid"/>
      <site site="tip"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="flexor_act" tendon="flexor" kp="40" ctrlrange="-0.2 0.2"/>
  </actuator>
</mujoco>
"""

# The same cable routed over a wrap geom that sits on a body carrying no site of
# its own, so the geom is OUTSIDE the span the sites alone describe. Rotating the
# wrist swings the anchor around that fixed sphere and so changes the cable's
# length: MuJoCo drives the wrist here, and a rule reading only the sites would
# miss it. In all 18 spatial drives shipped in the asset cache the wrap geom
# happens to sit on a body that already carries one of the route's sites, so the
# two rules coincide there and only a model like this one separates them.
WRAP_GEOM_XML = """
<mujoco model="pulley">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.04 0.04 0.01"/>
      <geom name="pulley_geom" type="sphere" size="0.008" pos="0 0 0.1"/>
      <site name="pulley_side" pos="0 0 0.112"/>
      <body name="palm" pos="0 0 0.06">
        <joint name="wrist" type="hinge" axis="0 1 0" range="-1 1" damping="0.05"/>
        <geom type="box" size="0.05 0.02 0.02"/>
        <site name="anchor" pos="-0.04 0 0.02"/>
        <body name="prox" pos="0.05 0 0">
          <joint name="mcp" type="hinge" axis="0 1 0" range="-1.6 1.6" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.012"/>
          <site name="tip" pos="0.06 0 0.015"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="flexor" width="0.002">
      <site site="anchor"/>
      <geom geom="pulley_geom" sidesite="pulley_side"/>
      <site site="tip"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="flexor_act" tendon="flexor" kp="40" ctrlrange="-0.2 0.2"/>
  </actuator>
</mujoco>
"""

# A route whose sites all sit on one link. Its length cannot change, so it
# drives nothing - the case a span rule must not over-report.
SAME_BODY_ROUTE_XML = """
<mujoco model="taut">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.04 0.04 0.01"/>
      <body name="link" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.012"/>
        <site name="near" pos="0.02 0 0.015"/>
        <site name="far" pos="0.08 0 0.015"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="taut" width="0.002">
      <site site="near"/>
      <site site="far"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="taut_act" tendon="taut" kp="40" ctrlrange="-0.2 0.2"/>
  </actuator>
</mujoco>
"""

# A cable between two separately floating hulls: the two routes share no
# ancestor but the world, so every joint on both walks is spanned.
TETHER_XML = """
<mujoco model="tether">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="hull_a" pos="0 0 0.5">
      <freejoint name="base_a"/>
      <geom type="sphere" size="0.05"/>
      <site name="sa" pos="0.05 0 0"/>
    </body>
    <body name="hull_b" pos="0.4 0 0.5">
      <freejoint name="base_b"/>
      <geom type="sphere" size="0.05"/>
      <site name="sb" pos="-0.05 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="tether" width="0.002">
      <site site="sa"/>
      <site site="sb"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="winch" tendon="tether" kp="40" ctrlrange="-0.2 0.2"/>
  </actuator>
</mujoco>
"""


def _physically_driven(model, act_id: int) -> frozenset[int]:
    """Joints whose generalized force responds to this actuator's ``ctrl``.

    MuJoCo's own answer to "what does this actuator move", and independent of
    every rule under test: the difference in ``qfrc_actuator`` between a scene at
    rest and the same scene with one ``ctrl`` displaced names the moved dofs, and
    ``dof_jntid`` maps those back to joints.
    """
    rest = mujoco.MjData(model)
    mujoco.mj_forward(model, rest)
    moved = mujoco.MjData(model)
    moved.ctrl[act_id] = 1.0
    mujoco.mj_forward(model, moved)
    delta = np.abs(np.asarray(moved.qfrc_actuator) - np.asarray(rest.qfrc_actuator))
    return frozenset(int(model.dof_jntid[v]) for v in np.nonzero(delta > 1e-9)[0])


def _ids(sim, obj, *names: str) -> frozenset[int]:
    model = sim._world._model
    out = set()
    for name in names:
        ident = mujoco.mj_name2id(model, obj, name)
        assert ident >= 0, name
        out.add(int(ident))
    return frozenset(out)


def _joint_value(sim, full_name: str) -> float:
    model, data = sim._world._model, sim._world._data
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
    assert jnt >= 0, full_name
    return float(data.qpos[int(model.jnt_qposadr[jnt])])


class TestTheReadingIsWhatMujocoMoves:
    """The reading is graded against ``qfrc_actuator``, not against the rule."""

    def test_a_cable_reports_the_joints_it_actually_drives(self, tmp_path: Path) -> None:
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fing/flexor_act")
            physical = _physically_driven(model, act)
            assert physical == _ids(sim, mujoco.mjtObj.mjOBJ_JOINT, "fing/mcp", "fing/pip"), (
                "premise: MuJoCo moves both finger joints from this one ctrl"
            )
            assert actuator_driven_joint_ids(model, act, mujoco) == physical
        finally:
            sim.cleanup()

    def test_a_wrap_geom_outside_the_sites_span_is_still_part_of_the_route(self, tmp_path: Path) -> None:
        """The obstacle the cable bends around moves it too, so it ends the span."""
        sim = _lone(tmp_path, "pul", WRAP_GEOM_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pul/flexor_act")
            physical = _physically_driven(model, act)
            both = _ids(sim, mujoco.mjtObj.mjOBJ_JOINT, "pul/wrist", "pul/mcp")
            assert physical == both, "premise: swinging the wrist past the sphere pulls the cable"
            assert actuator_driven_joint_ids(model, act, mujoco) == physical
        finally:
            sim.cleanup()

    def test_a_route_inside_one_link_drives_nothing(self, tmp_path: Path) -> None:
        """No joint can lengthen it, and MuJoCo agrees."""
        sim = _lone(tmp_path, "taut", SAME_BODY_ROUTE_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "taut/taut_act")
            assert _physically_driven(model, act) == frozenset(), "premise: nothing moves"
            assert actuator_driven_joint_ids(model, act, mujoco) == frozenset()
        finally:
            sim.cleanup()

    def test_a_tether_between_two_floating_hulls_reaches_both_bases(self, tmp_path: Path) -> None:
        """With no shared ancestor but the world, both walks are spanned."""
        sim = _lone(tmp_path, "teth", TETHER_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "teth/winch")
            physical = _physically_driven(model, act)
            assert physical == _ids(sim, mujoco.mjtObj.mjOBJ_JOINT, "teth/base_a", "teth/base_b"), (
                "premise: winching pulls on both floating bases"
            )
            assert actuator_driven_joint_ids(model, act, mujoco) == physical
        finally:
            sim.cleanup()


class TestTheSurfacesThatDriveTheRobot:
    """What the reading is for: commanding a joint, and refusing to double-drive it."""

    def test_a_cable_driven_joint_is_commandable_by_its_own_name(self, tmp_path: Path) -> None:
        """The #318 fallback: a joint name resolves to the actuator driving it."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            result = sim.send_action({"pip": 0.15}, robot_name="fing")
            assert result["status"] == "success", result["content"][0]["text"]
            model, data = sim._world._model, sim._world._data
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fing/flexor_act")
            assert float(data.ctrl[act]) != 0.0, "the value reached the cable's ctrl"
        finally:
            sim.cleanup()

    def test_commanding_the_cable_moves_the_finger(self, tmp_path: Path) -> None:
        """End to end: the joint the caller named actually flexes."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            start = _joint_value(sim, "fing/pip")
            assert sim.send_action({"pip": 0.18}, robot_name="fing")["status"] == "success"
            assert sim.step(1200)["status"] == "success"
            assert abs(_joint_value(sim, "fing/pip") - start) > 0.05
        finally:
            sim.cleanup()

    def test_actuate_robot_refuses_to_add_a_servo_beside_the_cable(self, tmp_path: Path) -> None:
        """The double-actuation the guard's own refusal exists to prevent."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            result = sim.actuate_robot(robot_name="fing")
            assert result["status"] == "error"
            text = result["content"][0]["text"]
            assert "flexor_act" in text
            assert "pip" in text and "mcp" in text
        finally:
            sim.cleanup()

    def test_a_cable_driven_joint_is_not_offered_as_a_pose_target(self, tmp_path: Path) -> None:
        """A cable's ctrl is in tendon units, so it lands in *other*, not *servos*."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            model = sim._world._model
            servos, other = joint_drive_map(model, mujoco)
            driven = _ids(sim, mujoco.mjtObj.mjOBJ_JOINT, "fing/mcp", "fing/pip")
            assert set(other) == set(driven)
            assert not set(servos) & set(driven)
        finally:
            sim.cleanup()

    def test_the_bodies_a_cable_moves_are_the_links_it_spans(self, tmp_path: Path) -> None:
        """The body question resolves through the same rule, so it agrees."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fing/flexor_act")
            assert actuator_target_body_ids(model, act, mujoco) == _ids(
                sim, mujoco.mjtObj.mjOBJ_BODY, "fing/prox", "fing/dist"
            )
        finally:
            sim.cleanup()


class TestWhatIsUnchanged:
    """Every reading the joint-wrap rule already got right is still that reading."""

    def test_a_fixed_tendon_still_reports_the_joints_it_lists(self, tmp_path: Path) -> None:
        sim = _lone(tmp_path, "grip", GRIPPER_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip/finger_act")
            wrapped = _ids(sim, mujoco.mjtObj.mjOBJ_JOINT, "grip/finger1", "grip/finger2")
            assert actuator_driven_joint_ids(model, act, mujoco) == wrapped
            assert _physically_driven(model, act) == wrapped
        finally:
            sim.cleanup()

    def test_a_site_transmission_still_commands_no_joint(self, tmp_path: Path) -> None:
        """A frame wrench moves a body without commanding a joint coordinate."""
        sim = _lone(tmp_path, "thruster", SITE_DRIVE_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thruster/thrust")
            assert actuator_driven_joint_ids(model, act, mujoco) == frozenset()
        finally:
            sim.cleanup()

    def test_the_per_ctrl_question_still_answers_minus_one_for_a_cable(self, tmp_path: Path) -> None:
        """A tendon is still not a joint, so no joint angle is written into it."""
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            model = sim._world._model
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fing/flexor_act")
            assert actuator_joint_id(model, act, mujoco) == -1
        finally:
            sim.cleanup()

    def test_a_stale_tendon_id_is_still_an_empty_answer(self, tmp_path: Path) -> None:
        sim = _lone(tmp_path, "fing", FLEXOR_XML)
        try:
            model = sim._world._model
            assert tendon_joint_ids(model, -1, mujoco) == frozenset()
            assert tendon_joint_ids(model, int(model.ntendon), mujoco) == frozenset()
        finally:
            sim.cleanup()
