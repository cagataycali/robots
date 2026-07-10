"""Regression tests for the ``base_velocity_tracking`` reward term.

The predicate/reward DSL grew ``base_velocity_tracking``: the bounded,
exponential-kernel counterpart of ``base_velocity`` and the canonical legged_gym
/ IsaacLab primary velocity-tracking reward
(``lin_weight * exp(-lin_err / sigma) + ang_weight * exp(-ang_err / sigma)``). It
consumes the same floating-base twist ``base_velocity`` does - ``base_lin_vel``
(world frame, rotated into the base frame via ``base_quat``) + ``base_ang_vel``
(already body-frame) - so the tracked velocity is heading-relative.

Unlike ``base_velocity`` (an UNBOUNDED negative-L2 error), this term is POSITIVE
and BOUNDED to ``[0, lin_weight + ang_weight]`` and peaks at perfect tracking -
the property that keeps the primary tracking reward well-scaled against the
bounded regularizer terms in an RL locomotion reward. These tests set a KNOWN
base pose + twist directly on a real MuJoCo sim (no mocks) and assert the
analytic kernel value, the bounded-max at perfect tracking, saturation to ~0 on
a large error, that planar and yaw tracking are weighted SEPARATELY, and that a
fixed-base arm degrades to 0.0. GL-free (get_observation with skip_images) so
they run in CI without a display.
"""

import logging
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import _reset_resolution_warnings, make_predicate, predicate_kind

# Floating base with a NAMED free joint plus one actuated hinge; get_observation
# surfaces base_pos/base_quat/base_lin_vel/base_ang_vel for this robot.
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

# Fixed-base arm: no free joint -> no base twist. The term must degrade to 0.0.
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


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_velocity_tracking", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_free_joint(sim, qpos7, qvel6):
    """Set the robot's (only) free joint pose + twist and forward the model."""
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 7] = qpos7
    data.qvel[vadr : vadr + 6] = qvel6
    mujoco.mj_forward(model, data)


# 90 deg about world +z; body +x points along world +y.
_Q_YAW90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
_LIN_WORLD = [1.1, 2.2, 3.3]  # base linear velocity in the WORLD frame
_ANG = [4.4, 5.5, 6.6]  # base angular velocity (body frame); yaw rate z = 6.6
# World->body rotation of _LIN_WORLD under _Q_YAW90 is [2.2, -1.1, 3.3], so the
# body-frame planar twist the reward tracks is (vx=2.2, vy=-1.1, wz=6.6).
_BODY_VX, _BODY_VY, _BODY_WZ = 2.2, -1.1, 6.6


def test_base_velocity_tracking_is_a_float_term():
    assert predicate_kind("base_velocity_tracking") == "float"


def test_peaks_at_lin_plus_ang_weight_on_perfect_tracking(sim):
    """Perfect tracking yields the bounded maximum lin_weight + ang_weight."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    # Defaults 1.0 / 0.5.
    assert make_predicate("base_velocity_tracking", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ)(sim) == pytest.approx(
        1.5, abs=1e-9
    )
    # Custom weights: peak is their sum.
    assert make_predicate(
        "base_velocity_tracking", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ, lin_weight=2.0, ang_weight=1.0
    )(sim) == pytest.approx(3.0, abs=1e-9)


def test_matches_analytic_exp_kernel(sim):
    """The value equals the sum of the two exponential kernels for a known error."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    # Planar error 0.5 in vx, no yaw error.
    cmd = dict(vx=_BODY_VX - 0.5, vy=_BODY_VY, wz=_BODY_WZ)
    expected = 1.0 * math.exp(-(0.5**2) / 0.25) + 0.5 * math.exp(-0.0 / 0.25)
    assert make_predicate("base_velocity_tracking", **cmd)(sim) == pytest.approx(expected, abs=1e-9)
    # Non-default sigma widens the kernel.
    exp_wide = 1.0 * math.exp(-(0.5**2) / 1.0) + 0.5
    assert make_predicate("base_velocity_tracking", tracking_sigma=1.0, **cmd)(sim) == pytest.approx(exp_wide, abs=1e-9)


def test_bounded_and_saturates_to_zero_on_large_error(sim):
    """The reward stays within [0, lin+ang] and decays to ~0 far from the command."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    far = make_predicate("base_velocity_tracking", vx=50.0, vy=50.0, wz=50.0)(sim)
    assert 0.0 <= far < 1e-6
    # A moderate mixed error stays strictly inside the (0, 1.5) open interval.
    mid = make_predicate("base_velocity_tracking", vx=0.0, vy=0.0, wz=0.0)(sim)
    assert 0.0 < mid < 1.5


def test_planar_and_yaw_are_weighted_separately(sim):
    """An equal-magnitude error costs more on the higher-weighted (planar) axis.

    With the default lin_weight=1.0 > ang_weight=0.5, a planar-only error of a
    given magnitude must reduce the reward MORE than a yaw-only error of the
    same magnitude - a distinction the single combined ``base_velocity`` norm
    (one weight for the whole twist) cannot express.
    """
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    e = 0.5
    planar_err = make_predicate("base_velocity_tracking", vx=_BODY_VX - e, vy=_BODY_VY, wz=_BODY_WZ)(sim)
    yaw_err = make_predicate("base_velocity_tracking", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ - e)(sim)
    # planar: 1.0*exp(-e^2/sigma) + 0.5 ; yaw: 1.0 + 0.5*exp(-e^2/sigma)
    assert planar_err == pytest.approx(1.0 * math.exp(-(e**2) / 0.25) + 0.5, abs=1e-9)
    assert yaw_err == pytest.approx(1.0 + 0.5 * math.exp(-(e**2) / 0.25), abs=1e-9)
    assert planar_err < yaw_err


def test_tracks_body_frame_not_world_frame(sim):
    """The world->body rotation fires: a world-frame command is not a perfect
    match under a rotated base, but the body-frame command is."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    # The world-frame planar command (1.1, 2.2) is NOT the body-frame twist, so
    # it does not peak.
    world_cmd = make_predicate("base_velocity_tracking", vx=1.1, vy=2.2, wz=6.6)(sim)
    assert world_cmd < 1.5
    # The body-frame command peaks.
    body_cmd = make_predicate("base_velocity_tracking", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ)(sim)
    assert body_cmd == pytest.approx(1.5, abs=1e-9)
    assert body_cmd > world_cmd


def test_non_positive_sigma_raises():
    with pytest.raises(ValueError, match="tracking_sigma"):
        make_predicate("base_velocity_tracking", tracking_sigma=0.0)
    with pytest.raises(ValueError, match="tracking_sigma"):
        make_predicate("base_velocity_tracking", tracking_sigma=-1.0)


def test_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base twist: the term degrades to 0.0 and warns."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_velocity_tracking", vx=0.5)(sim)
    assert val == 0.0
    assert any("base" in r.message.lower() for r in caplog.records)
