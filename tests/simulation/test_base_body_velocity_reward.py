"""Regression tests for the ``base_lin_vel_z`` / ``base_ang_vel_xy`` reward terms.

The predicate/reward DSL grew the two remaining pure-observation legged_gym /
IsaacLab locomotion regularizers - the ones that damp a floating base's
UNCOMMANDED velocity (both default-nonzero in ``LeggedRobotCfg``):

* ``base_lin_vel_z``  : ``-weight * v_body_z ** 2`` - penalise vertical bouncing.
* ``base_ang_vel_xy`` : ``-weight * (w_body_x ** 2 + w_body_y ** 2)`` - penalise
  roll/pitch wobble.

They complete the minimal velocity-tracking reward set alongside the previously
landed ``base_velocity`` (planar + yaw tracking), ``base_height`` (crouch
regularizer) and ``base_orientation`` (tilt regularizer): the position/orientation
terms penalise a static OFFSET, these two directly damp the RATE (a policy that
porpoises around the target height, or wobbles around level, averages ~0 offset
yet is caught here). Both read the base twist from ``get_observation``:
``base_lin_vel`` (world frame) rotated into the base frame via ``base_quat`` for
``v_body_z``; ``base_ang_vel`` (already body-frame) for ``w_body_x``/``w_body_y``.

These tests set a KNOWN base velocity directly on the sim (free-joint qvel:
world-frame linear + body-frame angular) and assert the correct negative squared
error, that ``base_lin_vel_z`` reads the BODY-frame vertical velocity (a
horizontal WORLD velocity under a 90-degree pitch is a vertical BODY velocity),
that ``base_ang_vel_xy`` is invariant to the yaw rate, and that both degrade to
``0.0`` (with a warning) on a fixed-base arm. They are GL-free (get_observation
with skip_images) so they run in CI without a display.
"""

import logging
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import (
    _reset_resolution_warnings,
    make_predicate,
)

# Floating base with a NAMED free joint (a humanoid's floating_base_joint) plus
# one actuated hinge. get_observation surfaces base_lin_vel/base_quat/base_ang_vel.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: no free joint anywhere -> no base twist. Both terms must
# degrade to 0.0 (and warn) rather than crash or invent a value.
FIXED_ARM_XML = """
<mujoco model="test_fixed">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


def _axis_quat(axis: str, deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a rotation of ``deg`` about a world axis."""
    h = math.radians(deg) / 2.0
    c, sn = math.cos(h), math.sin(h)
    return {
        "x": [c, sn, 0.0, 0.0],
        "y": [c, 0.0, sn, 0.0],
        "z": [c, 0.0, 0.0, sn],
    }[axis]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_body_velocity", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_vel(sim, quat_wxyz: list[float], lin_world: list[float], ang_body: list[float]) -> None:
    """Set the robot's (only) free joint to a fixed height + orientation and a
    known velocity. MuJoCo free-joint qvel is [linear (WORLD frame), angular
    (BODY frame)] - exactly what get_observation surfaces as base_lin_vel (world)
    and base_ang_vel (body)."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, 0.8, *quat_wxyz]
    data.qvel[vadr : vadr + 6] = [*lin_world, *ang_body]
    mujoco.mj_forward(model, data)


IDENT = [1.0, 0.0, 0.0, 0.0]


# --- base_lin_vel_z -------------------------------------------------------


def test_lin_vel_z_is_negative_squared_vertical_velocity(sim):
    """A level base moving at world (0.3,-0.4,0.5) has body vertical velocity 0.5:
    reward = -weight * 0.5**2."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_vel(sim, IDENT, [0.3, -0.4, 0.5], [0.0, 0.0, 0.0])
    assert make_predicate("base_lin_vel_z")(sim) == pytest.approx(-0.25, abs=1e-6)
    assert make_predicate("base_lin_vel_z", weight=4.0)(sim) == pytest.approx(-1.0, abs=1e-6)


def test_lin_vel_z_ignores_purely_horizontal_velocity_on_a_level_base(sim):
    """A level base sliding horizontally has zero vertical velocity -> 0 penalty
    (the term isolates the vertical component, not total speed)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_vel(sim, IDENT, [1.0, 2.0, 0.0], [0.0, 0.0, 0.0])
    assert make_predicate("base_lin_vel_z")(sim) == pytest.approx(0.0, abs=1e-6)


def test_lin_vel_z_uses_the_body_frame_not_the_world_frame(sim):
    """Under a 90-degree pitch the base's up-axis points along world +x, so a
    purely-horizontal WORLD velocity (0.6,0,0) is a vertical BODY velocity 0.6:
    reward = -0.6**2. This proves the term reads the base-frame vertical
    velocity, not world z."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_vel(sim, _axis_quat("y", 90.0), [0.6, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert make_predicate("base_lin_vel_z")(sim) == pytest.approx(-0.36, abs=1e-6)


def test_lin_vel_z_tracks_the_live_base_velocity(sim):
    """The reward reads the CURRENT base velocity: changing it changes the term."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_lin_vel_z")
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert term(sim) == pytest.approx(0.0, abs=1e-6)
    _set_base_vel(sim, IDENT, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    assert term(sim) == pytest.approx(-4.0, abs=1e-6)


def test_lin_vel_z_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base velocity: the term degrades to 0.0 and warns."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_lin_vel_z")(sim)
    assert val == 0.0
    assert any("base" in r.message.lower() for r in caplog.records)


# --- base_ang_vel_xy ------------------------------------------------------


def test_ang_vel_xy_is_negative_squared_roll_pitch_rate(sim):
    """Body angular velocity (0.1,0.2,0.3) has roll/pitch magnitude^2 0.01+0.04:
    reward = -(0.05); the yaw rate 0.3 is not penalised."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    assert make_predicate("base_ang_vel_xy")(sim) == pytest.approx(-0.05, abs=1e-6)
    assert make_predicate("base_ang_vel_xy", weight=4.0)(sim) == pytest.approx(-0.20, abs=1e-6)


def test_ang_vel_xy_is_invariant_to_yaw_rate(sim):
    """A pure yaw rate (turning in place) is NOT penalised -> 0. This is what
    makes it a roll/pitch-rate regularizer and not a turn penalty: a walking
    policy may turn freely, only rolling/pitching costs reward."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_ang_vel_xy")
    for wz in (0.5, -1.0, 3.0):
        _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.0, 0.0, wz])
        assert term(sim) == pytest.approx(0.0, abs=1e-6), f"yaw rate {wz} should not be penalised"


def test_ang_vel_xy_penalty_is_symmetric_roll_and_pitch(sim):
    """A pure roll rate is penalised identically to a pure pitch rate of the same
    magnitude (isotropic tip rate)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_ang_vel_xy")
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.7, 0.0, 0.0])
    roll = term(sim)
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.0, 0.7, 0.0])
    pitch = term(sim)
    assert roll == pytest.approx(pitch, abs=1e-6)
    assert roll == pytest.approx(-(0.7**2), abs=1e-6)


def test_ang_vel_xy_tracks_the_live_base_velocity(sim):
    """The reward reads the CURRENT base angular velocity."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_ang_vel_xy")
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert term(sim) == pytest.approx(0.0, abs=1e-6)
    _set_base_vel(sim, IDENT, [0.0, 0.0, 0.0], [0.3, 0.4, 0.0])
    assert term(sim) == pytest.approx(-0.25, abs=1e-6)


def test_ang_vel_xy_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base velocity: the term degrades to 0.0 and warns."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_ang_vel_xy")(sim)
    assert val == 0.0
    assert any("base" in r.message.lower() for r in caplog.records)
