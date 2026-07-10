"""Regression tests for the ``base_height`` locomotion-regularizer reward term.

The predicate/reward DSL grew a ``base_height`` reward term:
``-weight * (base_z - target) ** 2``, where ``base_z`` is a floating base's
world height from ``get_observation``'s ``base_pos`` signal. It is the standard
legged_gym / IsaacLab companion to ``base_velocity`` - a velocity-tracking
reward alone is degenerate (a policy can dive/crouch to cheat the forward
velocity), so a viable locomotion reward pairs the two in one ``dense_reward``
list (they sum per step).

These tests set a KNOWN base height directly on the sim and assert the term
computes the correct negative squared error - 0 at the target, symmetric above
and below, scaled by ``weight`` - and that it degrades to ``0.0`` (with a
warning) on a fixed-base arm that has no base position. They are GL-free
(get_observation with skip_images) so they run in CI without a display.
"""

import logging
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
# one actuated hinge. get_observation surfaces base_pos for this robot.
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

# Fixed-base arm: no free joint anywhere -> no base position. base_height must
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

_TARGET = 0.74  # a G1-pelvis-like target height


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_height", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_height(sim, z: float) -> None:
    """Set the robot's (only) free joint to an upright pose at world height z."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def test_base_height_is_zero_at_the_target(sim):
    """Reward is 0 when the base sits exactly at the target height."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_height(sim, _TARGET)
    assert make_predicate("base_height", target=_TARGET)(sim) == pytest.approx(0.0, abs=1e-6)


def test_base_height_is_negative_squared_error(sim):
    """Reward is -weight * (base_z - target)**2 when the base is off target."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_height(sim, 0.50)  # 0.24 m below the target
    expected = -((0.50 - _TARGET) ** 2)
    assert make_predicate("base_height", target=_TARGET)(sim) == pytest.approx(expected, abs=1e-6)
    # weight scales the penalty linearly.
    assert make_predicate("base_height", target=_TARGET, weight=5.0)(sim) == pytest.approx(5.0 * expected, abs=1e-6)


def test_base_height_penalty_is_symmetric_above_and_below(sim):
    """A base the same distance ABOVE the target is penalised identically to BELOW
    (the error is squared, not signed) - crouching and stilting cost the same."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_height", target=_TARGET)
    _set_base_height(sim, _TARGET - 0.24)
    below = term(sim)
    _set_base_height(sim, _TARGET + 0.24)
    above = term(sim)
    assert below == pytest.approx(above, abs=1e-6)
    assert below == pytest.approx(-(0.24**2), abs=1e-6)


def test_base_height_tracks_the_live_base_position(sim):
    """The reward reads the CURRENT base height: moving the base changes it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_height", target=_TARGET)
    _set_base_height(sim, _TARGET)
    assert term(sim) == pytest.approx(0.0, abs=1e-6)
    _set_base_height(sim, _TARGET - 0.3)
    assert term(sim) == pytest.approx(-(0.3**2), abs=1e-6)


def test_base_height_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base position: the term degrades to 0.0 and warns."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    # Reset the module-global warn-once dedup so this assertion is independent of
    # what other predicate tests warned first (the "robot base" key is shared).
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_height", target=0.5)(sim)
    assert val == 0.0
    assert any("base" in r.message.lower() for r in caplog.records)
