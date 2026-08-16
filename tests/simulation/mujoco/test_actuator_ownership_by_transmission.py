"""Which actuators a robot owns is decided by transmission, not by a raw id.

``SimRobot.actuator_ids`` is derived after every recompile by matching each
actuator against the robot. ``actuator_trnid[i, 0]`` holds a JOINT id only for a
``mjTRN_JOINT`` / ``mjTRN_JOINTINPARENT`` transmission; for a tendon, site,
slider-crank or body transmission it indexes a different table. Those id spaces
each start at 0, so comparing the raw value against a robot's ``joint_ids``
assigns an actuator to whichever robot happens to own the joint whose id equals
the other entity's.

A fixed tendon coupling a gripper's fingers is the standard MJCF idiom for a
coupled gripper, so the collision is reachable with ordinary two-robot scenes:
the gripper lands on the wrong robot and is missing from the one that carries
it, which makes it unoperable through ``set_gripper`` and lets the other robot
advertise an actuator that moves a different machine.

Every model here is inline MJCF loaded through ``add_robot(urdf_path=...)``, so
none of it downloads an asset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    actuator_joint_id,
    robot_owned_actuator_ids,
)

# A two-hinge arm driven by two direct joint-transmission position actuators.
ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.04 0.04 0.04"/>
      <body name="link1" pos="0 0 0.08">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.02"/>
        <body name="link2" pos="0.18 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.018"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" dampratio="1"/>
    <position name="elbow_act" joint="elbow" kp="30" dampratio="1"/>
  </actuator>
</mujoco>
"""

# The same arm with no actuators at all, to show a robot claiming an actuator
# it does not carry rather than merely being handed the wrong one.
UNACTUATED_ARM_XML = ARM_XML.replace(
    """  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" dampratio="1"/>
    <position name="elbow_act" joint="elbow" kp="30" dampratio="1"/>
  </actuator>
""",
    "",
)

# A gripper whose two fingers are coupled by one fixed tendon, driven by a
# single tendon-transmission actuator - the Panda-hand / Robotiq 2F-85 idiom.
# The ctrlrange is in tendon units, wider than any finger travel.
GRIPPER_XML = """
<mujoco model="gripper">
  <compiler angle="radian"/>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.03 0.03 0.02"/>
      <body name="finger_left" pos="0.02 0 0.03">
        <joint name="finger1" type="slide" axis="1 0 0" range="0 0.04" limited="true" damping="2"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
      <body name="finger_right" pos="-0.02 0 0.03">
        <joint name="finger2" type="slide" axis="-1 0 0" range="0 0.04" limited="true" damping="2"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip">
      <joint joint="finger1" coef="1"/>
      <joint joint="finger2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="finger_act" tendon="grip" kp="40" ctrlrange="0 255"/>
  </actuator>
</mujoco>
"""


# The ``sim`` parameter is left un-annotated in the helpers below: the engine
# types ``_world`` as ``SimWorld | None``, so an annotated helper would need a
# narrowing assert before every model read. Same convention as the sibling
# scene-mutation suites.


def _write(tmp_path: Path, name: str, xml: str) -> str:
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return str(path)


def _sim():
    from strands_robots.simulation import create_simulation

    return create_simulation("mujoco", tool_name="ownership_sim", mesh=False)


def _build(tmp_path: Path, order: tuple[str, ...], *, arm_xml: str = ARM_XML):
    """Build a scene holding an arm and a tendon gripper, added in ``order``."""
    sim = _sim()
    assert sim.create_world()["status"] == "success"
    xml_by_robot = {
        "arm": (_write(tmp_path, "arm.xml", arm_xml), [0.0, 0.0, 0.0]),
        "grip": (_write(tmp_path, "grip.xml", GRIPPER_XML), [0.5, 0.0, 0.0]),
    }
    for name in order:
        path, pos = xml_by_robot[name]
        assert sim.add_robot(name=name, urdf_path=path, position=pos)["status"] == "success"
    return sim


def _actuator_names(sim, robot_name: str) -> list[str]:
    model = sim._world._model
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in sim._world.robots[robot_name].actuator_ids
    ]


def _assert_collision_is_reproduced(sim, *, other: str) -> int:
    """Assert the fixture really does collide a tendon id with ``other``'s joint id.

    Without this the ownership assertions below could pass on a scene where the
    two id spaces never overlap, so nothing was being tested.
    """
    model = sim._world._model
    tendon_acts = [i for i in range(model.nu) if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_TENDON)]
    assert len(tendon_acts) == 1, tendon_acts
    act_id = tendon_acts[0]
    raw = int(model.actuator_trnid[act_id, 0])
    assert raw in set(sim._world.robots[other].joint_ids), (
        f"fixture does not reproduce the collision: tendon id {raw} is not a joint id of {other!r}"
    )
    return act_id


# -- ownership ---------------------------------------------------------------


@pytest.mark.parametrize("order", [("arm", "grip"), ("grip", "arm")], ids=["arm_first", "gripper_first"])
def test_each_robot_owns_exactly_its_own_actuators(tmp_path: Path, order: tuple[str, ...]) -> None:
    """A tendon-driven gripper belongs to the robot that carries it, either order."""
    sim = _build(tmp_path, order)
    try:
        _assert_collision_is_reproduced(sim, other="arm" if order[0] == "arm" else "grip")
        assert sorted(_actuator_names(sim, "arm")) == ["arm/elbow_act", "arm/shoulder_act"]
        assert _actuator_names(sim, "grip") == ["grip/finger_act"]
    finally:
        sim.cleanup()


def test_ownership_does_not_depend_on_the_order_robots_were_added(tmp_path: Path) -> None:
    """The same scene yields the same per-robot actuator names whatever the order.

    Pre-fix the gripper's owner was decided by which robot's joint ids happened
    to start at 0, so the two orders disagreed about who owns what.
    """
    by_order: dict[tuple[str, ...], dict[str, list[str]]] = {}
    for order in (("arm", "grip"), ("grip", "arm")):
        sim = _build(tmp_path, order)
        try:
            by_order[order] = {name: sorted(_actuator_names(sim, name)) for name in ("arm", "grip")}
        finally:
            sim.cleanup()
    assert by_order[("arm", "grip")] == by_order[("grip", "arm")]


def test_a_robot_declaring_no_actuators_owns_none(tmp_path: Path) -> None:
    """An unactuated robot must not be handed another robot's tendon actuator."""
    sim = _build(tmp_path, ("arm", "grip"), arm_xml=UNACTUATED_ARM_XML)
    try:
        _assert_collision_is_reproduced(sim, other="arm")
        assert sim._world.robots["arm"].actuator_ids == []
        assert _actuator_names(sim, "grip") == ["grip/finger_act"]
    finally:
        sim.cleanup()


def test_a_lone_robot_falls_back_to_every_actuator(tmp_path: Path) -> None:
    """The single-robot fallback is kept, and this is the case it exists for.

    A robot whose actuators are neither namespace-prefixed nor joint-driven
    matches neither ownership rule. In a one-robot scene every actuator is
    unambiguously that robot's, so the fallback claims them rather than leaving
    the robot with an empty action surface.
    """
    sim = _sim()
    try:
        assert sim.create_world()["status"] == "success"
        path = _write(tmp_path, "grip.xml", GRIPPER_XML)
        assert sim.add_robot(name="grip", urdf_path=path)["status"] == "success"
        robot = sim._world.robots["grip"]
        robot.namespace = ""  # emulate a scene loaded whole, with no attach prefix
        assert robot_owned_actuator_ids(sim._world._model, robot, mujoco) == []
        assert sim.reset()["status"] == "success"
        assert robot.actuator_ids == list(range(int(sim._world._model.nu)))
    finally:
        sim.cleanup()


def test_ownership_is_recomputed_when_a_robot_is_ejected(tmp_path: Path) -> None:
    """``remove_robot`` rebuilds the scene, and the surviving robot keeps its own."""
    sim = _build(tmp_path, ("grip", "arm"))
    try:
        assert sim.remove_robot("arm")["status"] == "success"
        assert _actuator_names(sim, "grip") == ["grip/finger_act"]
    finally:
        sim.cleanup()


# -- the capability the mis-assignment removed -------------------------------


@pytest.mark.parametrize("order", [("arm", "grip"), ("grip", "arm")], ids=["arm_first", "gripper_first"])
def test_set_gripper_operates_a_tendon_gripper_in_either_order(tmp_path: Path, order: tuple[str, ...]) -> None:
    """``set_gripper`` resolves its actuator from ``actuator_ids``, so it needs the right list.

    Pre-fix, with the arm's joint ids starting at 0 the gripper's actuator was
    absent from the gripper and ``set_gripper`` reported that the actuator did
    not exist in the model - blaming the model for an actuator that is in it.
    """
    sim = _build(tmp_path, order)
    try:
        model, data = sim._world._model, sim._world._data
        f1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grip/finger1")
        gap_before = float(data.qpos[model.jnt_qposadr[f1]])
        result = sim.set_gripper(robot_name="grip", state="open")
        assert result["status"] == "success", result["content"][0]["text"]
        gap_after = float(sim._world._data.qpos[model.jnt_qposadr[f1]])
        assert gap_after > gap_before + 0.005, (gap_before, gap_after)
    finally:
        sim.cleanup()


def test_actuate_robot_is_not_refused_by_a_tendon_id_collision(tmp_path: Path) -> None:
    """Adding servos to an unactuated robot is refused only if it really has some.

    The double-actuate guard compared every actuator's raw ``trnid`` against the
    robot's joint ids, so a tendon whose id matched one of them made the guard
    report actuators the robot does not have.
    """
    sim = _build(tmp_path, ("arm", "grip"), arm_xml=UNACTUATED_ARM_XML)
    try:
        _assert_collision_is_reproduced(sim, other="arm")
        nu_before = int(sim._world._model.nu)
        result = sim.actuate_robot(robot_name="arm", kp=40.0)
        assert result["status"] == "success", result["content"][0]["text"]
        assert int(sim._world._model.nu) == nu_before + 2
    finally:
        sim.cleanup()


def test_injected_position_actuators_are_owned_by_the_robot_that_asked(tmp_path: Path) -> None:
    """``actuate_robot`` names its actuators without the namespace prefix.

    They are owned via the joint they drive, which is why ownership is the union
    of the two rules rather than the namespace alone.
    """
    sim = _build(tmp_path, ("arm", "grip"), arm_xml=UNACTUATED_ARM_XML)
    try:
        assert sim.actuate_robot(robot_name="arm", kp=40.0)["status"] == "success"
        names = sorted(_actuator_names(sim, "arm"))
        assert names == ["arm_act_elbow", "arm_act_shoulder"], names
        assert not any(n.startswith("arm/") for n in names)
        assert _actuator_names(sim, "grip") == ["grip/finger_act"]
    finally:
        sim.cleanup()


def test_actuate_robot_holds_the_pose_without_seeding_a_tendon_drive(tmp_path: Path) -> None:
    """The hold-pose init reads a driven joint, so a tendon drive is left alone.

    Seeding it would write the gripper's setpoint from the angle of whatever
    joint the tendon's id collides with.
    """
    sim = _build(tmp_path, ("arm", "grip"), arm_xml=UNACTUATED_ARM_XML)
    try:
        model = sim._world._model
        grip_act = _assert_collision_is_reproduced(sim, other="arm")
        sim._world._data.ctrl[grip_act] = 7.0
        assert sim.set_joint_positions({"shoulder": 0.9}, robot_name="arm")["status"] == "success"
        assert sim.actuate_robot(robot_name="arm", kp=40.0)["status"] == "success"
        new_model = sim._world._model
        new_grip = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip/finger_act")
        assert float(sim._world._data.ctrl[new_grip]) == pytest.approx(7.0)
        shoulder_act = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_act_shoulder")
        assert float(sim._world._data.ctrl[shoulder_act]) == pytest.approx(0.9, abs=1e-3)
        assert model is not new_model
    finally:
        sim.cleanup()


# -- the shared gate ---------------------------------------------------------


def test_actuator_joint_id_resolves_a_joint_transmission(tmp_path: Path) -> None:
    """A joint-transmission actuator reports the joint it drives."""
    sim = _build(tmp_path, ("arm", "grip"))
    try:
        model = sim._world._model
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm/shoulder_act")
        jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm/shoulder")
        assert actuator_joint_id(model, act, mujoco) == jnt
    finally:
        sim.cleanup()


def test_actuator_joint_id_refuses_a_non_joint_transmission(tmp_path: Path) -> None:
    """A tendon drive reports no joint, even though its ``trnid`` is a valid joint id."""
    sim = _build(tmp_path, ("arm", "grip"))
    try:
        model = sim._world._model
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip/finger_act")
        raw = int(model.actuator_trnid[act, 0])
        assert 0 <= raw < int(model.njnt), "the raw id must look like a joint id for this to matter"
        assert actuator_joint_id(model, act, mujoco) == -1
    finally:
        sim.cleanup()
