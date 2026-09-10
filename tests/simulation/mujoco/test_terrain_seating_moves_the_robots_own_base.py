"""Regression tests: the shared free-base finder names the robot's OWN base.

``_robot_free_base_joint_id`` answers "which free joint is this robot's floating
base?" for two consumers: terrain seating (:meth:`_seat_floating_bases_on_terrain`,
which WRITES ``qpos`` at the joint it is given) and the ``start_recording``
schema (which only tests the answer against ``>= 0``).

The same question is answered a second and third time, inlined, by
``_get_sim_observation`` and ``get_robot_state``. Both of those spell it as
"the ownership-checked resolver decides; the named scan only supplies a
candidate", and both carry the same eleven-line comment explaining why: a robot
whose MJCF ships a free-jointed task object under its own namespace -- a
payload, a kick ball, a Menagerie grasping cube -- names that joint in
``robot.joint_names`` too, so a scan that CHOOSES from that list reports the
prop's pose as the robot's base.

The finder did not apply that precedence. It returned the first free joint named
in ``joint_names`` and only fell back to the ownership-checked resolver when the
scan found nothing, so for the shape below the two disagreed:

=========================================  ==========================
surface                                    base it named
=========================================  ==========================
``_get_sim_observation`` (ownership-first)  the chassis ``<freejoint>``
``get_robot_state`` (ownership-first)       the chassis ``<freejoint>``
``_robot_free_base_joint_id`` (scan-first)  ``payload_free``
=========================================  ==========================

The shape that separates them is a mobile base whose own free joint is UNNAMED
(LeKiwi's ``<freejoint/>``, absent from ``joint_names``) carrying a NAMED
free-jointed prop. The scan cannot see the real base at all and the prop is the
only free joint it can see, so it returned the prop; the ownership-checked walk
seeds from a declared wheel joint and reaches the real base by ancestry.

Terrain seating is where that becomes physical. Given the prop's joint it
samples the terrain height under the PROP and adds it to the PROP's ``z``, so
the robot keeps its flat-ground height -- buried under the raised terrain, the
one state the function exists to prevent -- while a task object levitates.

The two pre-existing pins on this finder assert only ``>= 0`` and
``jnt_type == mjJNT_FREE``. The prop's joint satisfies both, which is why the
disagreement was not observable before this module.

The controls below are what keep the fix from being an over-correction. Dropping
the named scan entirely (trusting ownership alone) also makes the three surfaces
agree, and is wrong: a fixed-base arm carrying a free-jointed prop has no owned
base, and the inlined loops deliberately let the scan's answer stand there
rather than erasing it ("Its ``-1`` is not allowed to erase a base the loop did
find"). ``test_a_fixed_base_arm_carrying_a_prop_keeps_the_scanned_joint`` is the
row that fails on that mutation.

The seating assertions are stated against the SAME robot spawned in a flat world
rather than against the MJCF's literal ``z``, so they measure the offset seating
applied and not the keyframe the model happens to compile to.
"""

import os
import tempfile
from typing import Any

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# A mobile base (LeKiwi-style): its own free joint is UNNAMED, so it is absent
# from ``robot.joint_names`` and only the kinematic-tree walk can reach it. The
# payload is a free-jointed task object shipped INSIDE the robot's own MJCF --
# how a Menagerie grasping scene is authored -- so its joint IS named in
# ``joint_names``. It sits far along +x, where the ``stairs`` terrain is a
# different height than under the chassis, so a seating pass that used it gets
# both the body and the magnitude wrong.
CARRIER_WITH_PROP_XML = """
<mujoco model="test_carrier_with_prop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="chassis" pos="0 0 0.15">
      <freejoint/>
      <geom name="chassis_geom" type="box" size="0.18 0.12 0.05" rgba="0.25 0.4 0.8 1" mass="8"/>
      <body name="wheel_l" pos="0 0.14 0">
        <joint name="wheel_left" type="hinge" axis="0 1 0"/>
        <geom name="wl" type="cylinder" size="0.06 0.02" quat="0.707 0.707 0 0" mass="0.5"/>
      </body>
      <body name="wheel_r" pos="0 -0.14 0">
        <joint name="wheel_right" type="hinge" axis="0 1 0"/>
        <geom name="wr" type="cylinder" size="0.06 0.02" quat="0.707 0.707 0 0" mass="0.5"/>
      </body>
    </body>
    <body name="payload" pos="3.5 0 0.06">
      <joint name="payload_free" type="free"/>
      <geom name="payload_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.5 0.1 1" mass="0.4"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="wheel_left_act" joint="wheel_left"/>
    <velocity name="wheel_right_act" joint="wheel_right"/>
  </actuator>
</mujoco>
"""

# Control: a humanoid-style NAMED floating base and no prop. Scan and ownership
# reach the same joint here, so this shape cannot separate them - it is what
# shows the fix did not stop resolving an ordinary floating base.
NAMED_BASE_XML = """
<mujoco model="test_named_base_only">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.05 0.2" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Control: a FIXED-base arm carrying a free-jointed prop. Ownership resolves
# nothing (no ancestor free joint from the declared joint or from the bodies its
# actuator drives), so the scan's answer is all there is. The inlined loops keep
# it rather than erasing it, and so must the finder.
FIXED_ARM_WITH_PROP_XML = """
<mujoco model="test_fixed_arm_with_prop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="column" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="link" pos="0 0 0.1">
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
    <body name="cube" pos="0.6 0 0.06">
      <joint name="cube_free" type="free"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.9 0.5 0.1 1" mass="0.3"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "model.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def terrain_sim():
    """A ``stairs`` world, so ``_seat_floating_bases_on_terrain`` is not a no-op."""
    s = Simulation(tool_name="test_terrain_seating_base", mesh=False)
    s.create_world(terrain="stairs", difficulty=1.0)
    yield s
    s.cleanup()


@pytest.fixture
def flat_sim():
    """A flat world: the seating pass returns immediately (``nhfield == 0``)."""
    s = Simulation(tool_name="test_terrain_seating_flat", mesh=False)
    s.create_world(ground_plane=True)
    yield s
    s.cleanup()


def _joint_id_by_type_scan(sim: Simulation, model: Any, want_named: str | None) -> int:
    """Locate a joint WITHOUT the finder under test.

    ``want_named=None`` asks for the scene's only UNNAMED free joint, which is
    the mobile base's own. Deriving the address from the finder would make every
    assertion below agree with whatever the finder returned.
    """
    mj = sim._mj
    found = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
        and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or "").endswith(want_named or "")
        and (want_named is not None or not mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i))
    ]
    assert len(found) == 1, f"expected exactly one free joint for {want_named!r}, got {found}"
    return found[0]


def _add_carrier(sim: Simulation) -> Any:
    assert sim.add_robot(name="carrier", urdf_path=_write(CARRIER_WITH_PROP_XML))["status"] == "success"
    world = sim._world
    assert world is not None, "premise: the fixture must have built a world"
    return world.robots["carrier"]


def test_the_prop_free_joint_is_named_in_joint_names(terrain_sim):
    """Premise. If the prop's joint were absent from ``joint_names`` the scan
    could not have chosen it and nothing below would grade anything."""
    robot = _add_carrier(terrain_sim)

    mj = terrain_sim._mj
    model = terrain_sim._world._model
    named_free = [
        n
        for n in robot.joint_names
        if model.jnt_type[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, (robot.namespace or "") + n)]
        == mj.mjtJoint.mjJNT_FREE
    ]

    assert named_free == ["payload_free"], (
        "premise: the prop must be the only free joint the named scan can see, and the robot's own "
        f"base must be invisible to it (an unnamed <freejoint>), got {named_free}"
    )


def test_the_world_carries_a_heightfield(terrain_sim):
    """Premise. On a flat plane the seating pass returns before reading the
    finder at all, so the seating cases would pass without the fix."""
    _add_carrier(terrain_sim)

    assert terrain_sim._world._model.nhfield == 1


def test_the_finder_names_the_robots_own_base_not_a_namespaced_prop(terrain_sim):
    """The core disagreement: the scan could only see the prop."""
    robot = _add_carrier(terrain_sim)
    model = terrain_sim._world._model
    own_base = _joint_id_by_type_scan(terrain_sim, model, None)
    prop = _joint_id_by_type_scan(terrain_sim, model, "payload_free")

    resolved = terrain_sim._robot_free_base_joint_id(model, robot)

    assert resolved != prop, "the finder named a namespaced task object as the robot's floating base"
    assert resolved == own_base


def test_the_finder_agrees_with_the_base_the_observation_reports(terrain_sim):
    """Cross-surface parity. ``get_observation`` already applies the
    ownership-checked precedence, so the finder disagreeing with it means one of
    the two is naming a joint that is not the robot's base."""
    robot = _add_carrier(terrain_sim)
    model = terrain_sim._world._model
    data = terrain_sim._world._data

    resolved = terrain_sim._robot_free_base_joint_id(model, robot)
    obs = terrain_sim.get_observation(robot_name="carrier", skip_images=True)

    adr = int(model.jnt_qposadr[resolved])
    assert obs["base_pos"] == pytest.approx([float(data.qpos[adr + i]) for i in range(3)], abs=1e-9)


def test_terrain_seating_raises_the_base_by_the_ground_beneath_it(terrain_sim, flat_sim):
    """The physical consequence. Measured against the same robot on flat ground,
    so this is the offset seating applied rather than a compiled keyframe."""
    _add_carrier(terrain_sim)
    model = terrain_sim._world._model
    data = terrain_sim._world._data
    base = _joint_id_by_type_scan(terrain_sim, model, None)
    adr = int(model.jnt_qposadr[base])

    _add_carrier(flat_sim)
    flat_model = flat_sim._world._model
    flat_adr = int(flat_model.jnt_qposadr[_joint_id_by_type_scan(flat_sim, flat_model, None)])
    flat_z = float(flat_sim._world._data.qpos[flat_adr + 2])

    ground = terrain_sim._ground_height_at(float(data.qpos[adr]), float(data.qpos[adr + 1]))
    assert ground > 0.0, "premise: the terrain under the chassis must be raised"
    assert float(data.qpos[adr + 2]) - flat_z == pytest.approx(ground, abs=1e-9), (
        "the robot kept its flat-ground height, so it is buried under the raised terrain"
    )


def test_terrain_seating_leaves_a_namespaced_prop_where_it_was(terrain_sim, flat_sim):
    """The other half: the offset was applied to the prop instead."""
    _add_carrier(terrain_sim)
    model = terrain_sim._world._model
    prop_adr = int(model.jnt_qposadr[_joint_id_by_type_scan(terrain_sim, model, "payload_free")])

    _add_carrier(flat_sim)
    flat_model = flat_sim._world._model
    flat_prop_adr = int(flat_model.jnt_qposadr[_joint_id_by_type_scan(flat_sim, flat_model, "payload_free")])

    assert float(terrain_sim._world._data.qpos[prop_adr + 2]) == pytest.approx(
        float(flat_sim._world._data.qpos[flat_prop_adr + 2]), abs=1e-9
    ), "seating moved a task object instead of the robot's base"


def test_a_named_floating_base_with_no_prop_is_still_resolved(terrain_sim):
    """Control. Scan and ownership agree on this shape, so it holds either way -
    it is what shows the fix did not stop resolving an ordinary floating base."""
    assert terrain_sim.add_robot(name="hum", urdf_path=_write(NAMED_BASE_XML))["status"] == "success"
    model = terrain_sim._world._model
    robot = terrain_sim._world.robots["hum"]

    resolved = terrain_sim._robot_free_base_joint_id(model, robot)

    assert resolved == _joint_id_by_type_scan(terrain_sim, model, "floating_base_joint")


def test_a_fixed_base_arm_carrying_a_prop_keeps_the_scanned_joint(terrain_sim):
    """Control for the over-correction. Ownership resolves nothing here, and the
    inlined loops keep what the scan found rather than erasing it, so the finder
    must too. Trusting ownership alone would return ``-1`` and fail this."""
    assert terrain_sim.add_robot(name="arm", urdf_path=_write(FIXED_ARM_WITH_PROP_XML))["status"] == "success"
    model = terrain_sim._world._model
    robot = terrain_sim._world.robots["arm"]

    resolved = terrain_sim._robot_free_base_joint_id(model, robot)

    assert resolved == _joint_id_by_type_scan(terrain_sim, model, "cube_free")


@pytest.mark.parametrize(
    "name,xml",
    [
        ("carrier", CARRIER_WITH_PROP_XML),
        ("hum", NAMED_BASE_XML),
        ("arm", FIXED_ARM_WITH_PROP_XML),
    ],
)
def test_the_recording_schema_predicate_tracks_the_observations_base(terrain_sim, name, xml):
    """Control for the second consumer. ``start_recording`` declares the 13 base
    columns on ``finder(...) >= 0``, and ``get_observation`` fills them, so the
    two must agree on WHETHER there is a base. They already did - the finder and
    the loops can only differ on WHICH joint, never on whether one exists - and
    this pins that the fix did not introduce a gap between them."""
    assert terrain_sim.add_robot(name=name, urdf_path=_write(xml))["status"] == "success"
    robot = terrain_sim._world.robots[name]

    has_base_column = terrain_sim._robot_free_base_joint_id(terrain_sim._world._model, robot) >= 0
    obs = terrain_sim.get_observation(robot_name=name, skip_images=True)

    assert has_base_column == ("base_pos" in obs)
