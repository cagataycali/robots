"""Regression tests for the ``base_orientation`` locomotion-regularizer reward term.

The predicate/reward DSL grew a ``base_orientation`` reward term:
``-weight * (g_x ** 2 + g_y ** 2)``, where ``(g_x, g_y, g_z)`` is the world
gravity direction expressed in the floating base's frame (the "projected
gravity" a legged controller reads), derived from ``get_observation``'s
``base_quat`` signal. It is the standard legged_gym / IsaacLab ``orientation``
regularizer and the third piece of a minimal velocity-tracking reward:
``base_velocity`` alone is degenerate (a policy can crouch OR lean to cheat the
forward velocity), so a viable locomotion reward pairs ``base_velocity`` with
``base_height`` (stops crouch-cheating) AND ``base_orientation`` (stops
lean/tilt-cheating) in one ``dense_reward`` list (they sum per step).

These tests set a KNOWN base orientation directly on the sim and assert the
term computes the correct negative squared projected-gravity error - 0 when
level, ``-sin(theta) ** 2`` at a roll/pitch of ``theta``, symmetric across roll
and pitch, INVARIANT to yaw (heading), scaled by ``weight`` - and that it
degrades to ``0.0`` (with a warning) on a fixed-base arm that has no base
orientation. They are GL-free (get_observation with skip_images) so they run in
CI without a display.
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
# one actuated hinge. get_observation surfaces base_quat for this robot.
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

# Fixed-base arm: no free joint anywhere -> no base orientation. base_orientation
# must degrade to 0.0 (and warn) rather than crash or invent a value.
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
    s = Simulation(tool_name="test_base_orientation", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_quat(sim, quat_wxyz: list[float]) -> None:
    """Set the robot's (only) free joint to a fixed world height with orientation quat."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, 0.8, *quat_wxyz]
    mujoco.mj_forward(model, data)


def test_base_orientation_is_zero_when_level(sim):
    """Reward is 0 when the base is perfectly level (projected gravity = (0,0,-1))."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_quat(sim, [1.0, 0.0, 0.0, 0.0])
    assert make_predicate("base_orientation")(sim) == pytest.approx(0.0, abs=1e-6)


def test_base_orientation_is_negative_squared_projected_gravity(sim):
    """A roll of theta gives projected-gravity xy magnitude sin(theta):
    reward = -weight * sin(theta)**2."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_quat(sim, _axis_quat("x", 30.0))
    expected = -(math.sin(math.radians(30.0)) ** 2)  # = -0.25
    assert make_predicate("base_orientation")(sim) == pytest.approx(expected, abs=1e-6)
    # weight scales the penalty linearly.
    assert make_predicate("base_orientation", weight=4.0)(sim) == pytest.approx(4.0 * expected, abs=1e-6)


def test_base_orientation_penalty_is_symmetric_roll_and_pitch(sim):
    """A pitch of theta is penalised identically to a roll of theta (isotropic tilt)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_orientation")
    _set_base_quat(sim, _axis_quat("x", 20.0))
    roll = term(sim)
    _set_base_quat(sim, _axis_quat("y", 20.0))
    pitch = term(sim)
    assert roll == pytest.approx(pitch, abs=1e-6)
    assert roll == pytest.approx(-(math.sin(math.radians(20.0)) ** 2), abs=1e-6)


def test_base_orientation_is_invariant_to_yaw(sim):
    """Pure yaw (heading change) keeps the base level -> zero penalty. This is what
    makes it a flat-orientation regularizer and not a heading penalty: a walking
    policy may turn freely, only roll/pitch off level costs reward."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_orientation")
    for yaw in (45.0, 90.0, 179.0):
        _set_base_quat(sim, _axis_quat("z", yaw))
        assert term(sim) == pytest.approx(0.0, abs=1e-6), f"yaw {yaw} should not be penalised"


def test_base_orientation_tracks_the_live_base_orientation(sim):
    """The reward reads the CURRENT base orientation: tipping the base changes it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_orientation")
    _set_base_quat(sim, [1.0, 0.0, 0.0, 0.0])
    assert term(sim) == pytest.approx(0.0, abs=1e-6)
    _set_base_quat(sim, _axis_quat("y", 45.0))
    assert term(sim) == pytest.approx(-(math.sin(math.radians(45.0)) ** 2), abs=1e-6)


def test_base_orientation_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base orientation: the term degrades to 0.0 and warns."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    # Reset the module-global warn-once dedup so this assertion is independent of
    # what other predicate tests warned first (the "robot base" key is shared).
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_orientation")(sim)
    assert val == 0.0
    assert any("base" in r.message.lower() for r in caplog.records)
