"""Regression tests: a floating-base robot spawns SEATED on the local terrain.

``create_world(terrain=...)`` (#1336/#1338/#1339/#1340) lays a heightfield so a
locomotion robot can be spawned and evaluated on non-flat ground -- that is the
feature's stated purpose. But a robot's model spawns its free base at the
flat-ground keyframe height (e.g. the Unitree Go2 base at ``z=0.445``, feet
~``z=0.02``), which on a raised heightfield leaves the feet BELOW the terrain
surface: the robot spawns *buried* in the ground, with penetration that grows
with the curriculum ``difficulty``. Worse, a naive one-shot fix in ``add_robot``
would not survive ``reset()`` (which ``run_policy`` / ``eval_policy`` call before
and between episodes), snapping the base back to the buried flat-ground height.

The engine now seats every floating base on the local terrain -- offsetting its
``z`` by ``_ground_height_at(x, y)`` -- at ``add_robot`` spawn AND on every
``reset()``, so the robot rests on the surface (feet just clear of it) at the
start of every episode. A flat ground plane is a no-op (the height is ``0.0``)
and a fixed-base arm (no free joint) is skipped. These tests are GL-free
(``mesh=False``, no render) so they run in CI without a display.
"""

import os
import tempfile

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.terrain import terrain_elevation  # noqa: E402

# A LOW floating base (base at z=0.4, foot sphere bottom at z=0.02) -- like a
# quadruped, its feet sit well below a raised heightfield peak at the flat
# spawn, so the burying is unambiguous. Two variants exercise the NAMED-free-
# joint (humanoid ``floating_base_joint``) and UNNAMED-``<freejoint>`` (mobile
# base, e.g. Go2) detection paths.
_NAMED_LOW_BASE = """
<mujoco model="seat_named">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.4">
      <freejoint name="floating_base_joint"/>
      <geom name="torso" type="box" size="0.1 0.05 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="leg" pos="0 0 -0.35">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="foot" type="sphere" size="0.03" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="knee_act" joint="knee"/></actuator>
</mujoco>
"""

_UNNAMED_LOW_BASE = _NAMED_LOW_BASE.replace('<freejoint name="floating_base_joint"/>', "<freejoint/>").replace(
    "seat_named", "seat_unnamed"
)

# A fixed-base arm (no free joint): the seat must skip it without error.
_FIXED_ARM = """
<mujoco model="seat_fixed_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link0" pos="0 0 0.0">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="j0" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="j0_act" joint="j0"/></actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


# ``model.geom_type[i]`` is a numpy integer, so it is compared to the geom-type
# enum BY VALUE: ``x in (enum, ...)`` puts the enum on the left of ``==``, where a
# pybind11 enum does not match a numpy integer on every mujoco build.
_GROUND_GEOM_TYPES = (int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_PLANE))


def _lowest_collision_geom_z(sim) -> float:
    """World z of the lowest collidable robot geom (skips the ground plane / hfield)."""
    model, data = sim._world._model, sim._world._data
    zs = []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        if int(model.geom_type[gid]) in _GROUND_GEOM_TYPES:
            continue
        zs.append(float(data.geom_xpos[gid][2]))
    assert zs, "no collidable robot geoms found"
    return min(zs)


def _base_z(sim) -> float:
    model, data = sim._world._model, sim._world._data
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return float(data.qpos[int(model.jnt_qposadr[j]) + 2])
    raise AssertionError("no free joint")


@pytest.mark.parametrize("xml", [_NAMED_LOW_BASE, _UNNAMED_LOW_BASE], ids=["named_base", "unnamed_base"])
def test_floating_base_spawns_seated_on_terrain(xml):
    """A floating base spawns with its feet ON the terrain, not buried below it."""
    sim = Simulation(tool_name="seat_terrain_spawn", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("floater", urdf_path=_write(xml))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert ground > 0.1  # a genuinely raised plateau (pyramid peak at the centre)
        # The robot's flat-spawn feet (~0.02) would sit BELOW this plateau; the
        # seat must have raised the base so the lowest geom clears the surface.
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6
    finally:
        sim.cleanup()


def test_terrain_seat_survives_reset():
    """The seat is re-applied on reset() -- run_policy/eval_policy reset each episode."""
    sim = Simulation(tool_name="seat_terrain_reset", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6  # seated at spawn
        assert sim.reset()["status"] == "success"
        # After reset the base must still be seated (a one-shot add_robot fix
        # would snap back to the buried flat-ground height here).
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6
    finally:
        sim.cleanup()


def test_seat_offset_scales_with_difficulty():
    """The base is raised by exactly the local terrain height (curriculum knob)."""
    for difficulty in (0.5, 1.0, 2.0):
        sim = Simulation(tool_name="seat_terrain_diff", mesh=False)
        sim.create_world(ground_plane=True, terrain="pyramid", difficulty=difficulty)
        sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
        try:
            ground = sim._ground_height_at(0.0, 0.0)
            assert ground == pytest.approx(terrain_elevation(difficulty), abs=1e-6)
            # base z == flat spawn (0.4) + the local terrain height.
            assert _base_z(sim) == pytest.approx(0.4 + ground, abs=1e-6)
        finally:
            sim.cleanup()


def test_flat_ground_spawn_is_unchanged():
    """No heightfield -> the seat is a no-op (byte-for-byte flat behaviour)."""
    sim = Simulation(tool_name="seat_flat_noop", mesh=False)
    sim.create_world(ground_plane=True)
    sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
    try:
        assert sim._ground_height_at(0.0, 0.0) == 0.0
        assert _base_z(sim) == pytest.approx(0.4, abs=1e-9)  # flat keyframe z, no offset
        sim.reset()
        assert _base_z(sim) == pytest.approx(0.4, abs=1e-9)
    finally:
        sim.cleanup()


def test_fixed_base_arm_on_terrain_is_skipped():
    """A fixed-base arm has no free joint -> the seat skips it without error."""
    sim = Simulation(tool_name="seat_fixed_arm", mesh=False)
    sim.create_world(ground_plane=True, terrain="rough", difficulty=1.0)
    result = sim.add_robot("arm", urdf_path=_write(_FIXED_ARM))
    try:
        assert result["status"] == "success"
        assert sim.reset()["status"] == "success"  # no crash on the free-joint-less arm
    finally:
        sim.cleanup()


# Three shapes the height sample under the BASE cannot seat, one mechanism each.
#
# 1. A flat pose that does not clear ``z=0``: the foot hangs 50 mm BELOW the
#    plane the model is authored against. Real assets are built this way -
#    LeKiwi's wheels sit 34.6 mm under its root body (its own scene recesses the
#    floor to -0.1 m) - and the offset carries that burial onto the terrain.
_BURIED_FLAT_POSE = """
<mujoco model="seat_buried_flat">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.05">
      <freejoint name="floating_base_joint"/>
      <geom name="torso" type="box" size="0.1 0.05 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="leg" pos="0 0 -0.05">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="foot" type="sphere" size="0.03" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="knee_act" joint="knee"/></actuator>
</mujoco>
"""

# 2. A base standing on feet 0.3 m out to either side, whose flat pose DOES
#    clear ``z=0``: on a bumpy heightfield the surface under a foot is not the
#    surface under the base, so an offset read under the base alone drives the
#    uphill foot in. Spawned off-centre, where ``rough`` slopes that way.
_WIDE_STANCE = """
<mujoco model="seat_wide_stance">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.4">
      <freejoint name="floating_base_joint"/>
      <geom name="torso" type="box" size="0.1 0.05 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="leg_front" pos="0.3 0 -0.36">
        <joint name="knee_front" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="foot_front" type="sphere" size="0.03" rgba="0.8 0.3 0.3 1"/>
      </body>
      <body name="leg_back" pos="-0.3 0 -0.36">
        <joint name="knee_back" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="foot_back" type="sphere" size="0.03" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_front_act" joint="knee_front"/>
    <motor name="knee_back_act" joint="knee_back"/>
  </actuator>
</mujoco>
"""

# 3. A leg reaching 180 mm below the base - the Unitree A1's straight-legged
#    spawn (feet 120 mm under ``z=0``) - so the shin ends up INSIDE the
#    heightfield prism and MuJoCo resolves it SIDEWAYS: measured ``dist``
#    -25.0 mm with ``normal_z`` 0.000 while the surface stands 84.5 mm above the
#    contact point. A seat that lifts by the penetration depth alone moves it
#    25 mm and leaves it buried, which is the row that grades the second term.
_DEEP_LEG = """
<mujoco model="seat_deep_leg">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.05">
      <freejoint name="floating_base_joint"/>
      <geom name="torso" type="box" size="0.1 0.05 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="leg" pos="0 0 -0.18">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="shin" type="capsule" fromto="0 0 0 0 0 0.14" size="0.025" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="knee_act" joint="knee"/></actuator>
</mujoco>
"""


def _deepest_ground_penetration(sim) -> float:
    """Depth (m, ``>= 0``) the robot's own tree is inside the ground geoms.

    Read off MuJoCo's contact list rather than off the seat's own helper, so the
    assertions grade the physical state and not the arithmetic that produced it.
    Scoped by ``body_rootid`` to the tree the free base moves.
    """
    model, data = sim._world._model, sim._world._data
    mujoco.mj_forward(model, data)
    base = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    root = int(model.body_rootid[model.jnt_bodyid[base]])
    deepest = 0.0
    for con in data.contact[: int(data.ncon)]:
        pair = (int(con.geom1), int(con.geom2))
        ground = [g for g in pair if int(model.geom_type[g]) in _GROUND_GEOM_TYPES]
        if len(ground) != 1:
            continue
        other = pair[1] if ground[0] == pair[0] else pair[0]
        if int(model.body_rootid[model.geom_bodyid[other]]) != root:
            continue
        deepest = min(deepest, float(con.dist))
    return -deepest


@pytest.mark.parametrize(
    "xml,terrain,position",
    [
        (_BURIED_FLAT_POSE, "pyramid", None),
        (_WIDE_STANCE, "rough", [1.0, 0.0, 0.0]),
        (_DEEP_LEG, "rough", None),
    ],
    ids=["buried_flat_pose", "wide_stance_on_bumps", "leg_inside_the_prism"],
)
def test_nothing_of_a_seated_robot_is_left_inside_the_terrain(xml, terrain, position):
    """The seat is MEASURED, not assumed: no geom is left inside the ground.

    The heightfield height beneath the base answers none of the three shapes
    above, so each spawns buried under a height-sample-only seat - the one state
    this seat exists to prevent - by 30.0 mm, 11.1 mm and 25.0 mm respectively.
    """
    sim = Simulation(tool_name="seat_measured_spawn", mesh=False)
    sim.create_world(ground_plane=True, terrain=terrain, difficulty=2.0)
    sim.add_robot("floater", urdf_path=_write(xml), position=position)
    try:
        assert _deepest_ground_penetration(sim) == pytest.approx(0.0, abs=1e-3)
        assert sim.reset()["status"] == "success"
        assert _deepest_ground_penetration(sim) == pytest.approx(0.0, abs=1e-3)
    finally:
        sim.cleanup()
