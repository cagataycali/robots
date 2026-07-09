"""Cross-backend correctness of the locomotion reward terms (MuJoCo == Newton).

The predicate/reward DSL grew the minimal legged_gym / IsaacLab velocity-tracking
reward set - ``base_velocity`` (heading-relative twist tracking), ``base_height``
(crouch regularizer), ``base_orientation`` (flat-orientation regularizer) and the
two uncommanded-base-velocity regularizers ``base_lin_vel_z`` (anti-bounce) and
``base_ang_vel_xy`` (anti-wobble). All read a floating base's 6-DoF pose + twist
from ``get_observation``
(``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel``), and their
correctness rests on a fragile, hard-won FRAME contract that must hold IDENTICALLY
on both simulation backends:

* ``base_lin_vel`` is WORLD-frame on both backends (``base_velocity`` rotates it
  into the base frame via ``base_quat`` to get the heading-relative vx/vy);
* ``base_ang_vel`` is BODY-frame on both backends (the IMU-gyro convention);
* ``base_quat`` is ``[w, x, y, z]`` on both backends.

The per-backend reward tests build only a MuJoCo sim, and the Newton floating-base
tests assert the raw surfacing structure - but nothing had ever evaluated the
reward TERMS through the Newton engine, nor pinned that the SAME physical base
state yields the SAME reward on both backends. A frame flip or a dropped base key
on either backend (the class fixed for Newton's world->body angular velocity and
for the MuJoCo base linear-velocity surfacing) would silently corrupt every
locomotion reward on that backend with no existing test failing.

These tests set an IDENTICAL known base state on a real MuJoCo Simulation AND a
real Newton engine (no mocks) and assert each reward term equals its analytic
value on BOTH backends and that the two backends agree. They are GL-free
(``get_observation(skip_images=True)``) so they run in CI without a display; the
Newton half skips when newton/warp are not installed.
"""

import importlib.util
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import _reset_resolution_warnings, make_predicate

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

# A minimal floating-base robot buildable on BOTH backends from one source: a
# NAMED free-jointed root (a humanoid's ``floating_base_joint``) carrying two
# hinge children. Geoms carry mass so the Newton solver accepts the model.
_FLOATER_MJCF = """<mujoco model="floater">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="pelvis" pos="0 0 0.5">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

# A fixed-base arm (no free joint) for the graceful-degradation parity check.
_FIXED_ARM_MJCF = """<mujoco model="arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>"""


def _yaw_quat(deg: float) -> list[float]:
    """(w, x, y, z) quaternion for a rotation of ``deg`` about the world +z axis."""
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _roll_quat(deg: float) -> list[float]:
    """(w, x, y, z) quaternion for a rotation of ``deg`` about the world +x axis."""
    h = math.radians(deg) / 2.0
    return [math.cos(h), math.sin(h), 0.0, 0.0]


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


# ---- backend builders + known-state setters ---------------------------------
#
# Each backend stores a free joint's state in its own layout, so setting the
# SAME physical state takes a backend-specific write. The pure-z angular velocity
# used below maps identically (a spin about world +z equals a spin about body +z
# under a yaw rotation), so ``ang`` is written verbatim on both.


def _build_mujoco(xml: str) -> Simulation:
    sim = Simulation(tool_name="test_xback", mesh=False)
    sim.create_world(ground_plane=False)
    assert sim.add_robot("floater", urdf_path=_write(xml))["status"] == "success"
    return sim


def _set_mujoco(sim: Simulation, quat_wxyz: list[float], z: float, lin_world: list[float], ang: list[float]) -> None:
    assert sim._world is not None and sim._world._model is not None and sim._world._data is not None
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, z, *quat_wxyz]
    # MuJoCo free-joint qvel is [linear WORLD, angular BODY].
    data.qvel[dadr : dadr + 6] = [*lin_world, *ang]
    mujoco.mj_forward(model, data)


def _build_newton(xml: str):
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    eng = NewtonSimEngine(solver="mujoco")
    eng.create_world()
    assert eng.add_robot("floater", urdf_path=_write(xml))["status"] == "success"
    return eng


def _set_newton(eng, quat_wxyz: list[float], z: float, lin_world: list[float], ang: list[float]) -> None:
    bj = eng._robot_free_base_joint["floater"]
    q_idx = eng._joint_coord_index[("floater", bj)]
    d_idx = eng._joint_dof_index[("floater", bj)]
    q = eng._state_0.joint_q.numpy().copy()
    q[q_idx : q_idx + 3] = [0.0, 0.0, z]
    # Newton stores the free-joint quaternion as xyzw.
    w, x, y, zc = quat_wxyz
    q[q_idx + 3 : q_idx + 7] = [x, y, zc, w]
    eng._state_0.joint_q.assign(q)
    qd = eng._state_0.joint_qd.numpy().copy()
    qd[:] = 0.0
    # Newton free-joint qd is [linear WORLD, angular WORLD]; a pure-z spin is
    # frame-invariant under a yaw rotation, so the value matches MuJoCo's body ang.
    qd[d_idx : d_idx + 6] = [*lin_world, *ang]
    eng._state_0.joint_qd.assign(qd)


def _base_obs(sim) -> dict:
    obs = sim.get_observation(robot_name="floater", skip_images=True)
    return {k: obs.get(k) for k in ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")}


# ---- the moving-base state used by the main cross-backend test ---------------
# Base yawed +90 about z at height 0.8, gliding at 1 m/s along WORLD +x while
# spinning at 2 rad/s about z. In the base frame the world +x velocity is body
# -y, so base_velocity(0,0,0) tracks (vx=0, vy=-1, wz=2) -> -sqrt(5). The base is
# level (pure yaw) so base_orientation is 0.
_QUAT = _yaw_quat(90.0)
_Z = 0.8
_LIN = [1.0, 0.0, 0.0]
_ANG = [0.0, 0.0, 2.0]
_EXP_VELOCITY = -math.sqrt(5.0)  # -||(0, -1, 2)||
_EXP_HEIGHT = -((_Z - 0.5) ** 2)  # -(0.3)^2 = -0.09
_EXP_ORIENT = 0.0


def _assert_moving_base_rewards(sim, tag: str) -> None:
    base = _base_obs(sim)
    assert base["base_pos"] == pytest.approx([0.0, 0.0, _Z], abs=1e-4), tag
    assert base["base_quat"] == pytest.approx(_QUAT, abs=1e-4), tag
    assert base["base_lin_vel"] == pytest.approx(_LIN, abs=1e-4), tag
    assert base["base_ang_vel"] == pytest.approx(_ANG, abs=1e-4), tag
    assert make_predicate("base_velocity", vx=0.0, vy=0.0, wz=0.0)(sim) == pytest.approx(_EXP_VELOCITY, abs=1e-4), tag
    assert make_predicate("base_height", target=0.5)(sim) == pytest.approx(_EXP_HEIGHT, abs=1e-4), tag
    assert make_predicate("base_orientation")(sim) == pytest.approx(_EXP_ORIENT, abs=1e-4), tag


def test_moving_base_rewards_correct_on_mujoco():
    """All three terms compute the analytic reward for a known moving base (MuJoCo)."""
    sim = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(sim, _QUAT, _Z, _LIN, _ANG)
    _assert_moving_base_rewards(sim, "mujoco")
    sim.cleanup()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_moving_base_rewards_correct_on_newton():
    """The SAME terms compute the SAME analytic reward through the Newton engine.

    First test to evaluate the locomotion reward terms through Newton at all -
    it exercises the full get_observation -> reward pipeline, so a Newton base
    frame/key regression fails here even though the MuJoCo reward tests pass.
    """
    eng = _build_newton(_FLOATER_MJCF)
    _set_newton(eng, _QUAT, _Z, _LIN, _ANG)
    _assert_moving_base_rewards(eng, "newton")
    eng.destroy()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_moving_base_rewards_identical_across_backends():
    """MuJoCo and Newton must agree exactly for the same physical base state.

    This is the guard the per-backend tests cannot provide: the SAME reward term
    fed the SAME base state must return the SAME value on both engines. A silent
    frame flip or dropped base key on either backend makes the two diverge here.
    """
    mj = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(mj, _QUAT, _Z, _LIN, _ANG)
    nt = _build_newton(_FLOATER_MJCF)
    _set_newton(nt, _QUAT, _Z, _LIN, _ANG)

    mj_base, nt_base = _base_obs(mj), _base_obs(nt)
    for key in ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel"):
        assert nt_base[key] == pytest.approx(mj_base[key], abs=1e-4), key

    for factory, kwargs in (
        ("base_velocity", {"vx": 0.0, "vy": 0.0, "wz": 0.0}),
        ("base_velocity", {"vx": 0.5, "vy": -1.0, "wz": 2.0}),  # non-zero target too
        ("base_height", {"target": 0.5}),
        ("base_orientation", {"weight": 2.0}),
    ):
        mj_r = make_predicate(factory, **kwargs)(mj)
        nt_r = make_predicate(factory, **kwargs)(nt)
        assert nt_r == pytest.approx(mj_r, abs=1e-4), f"{factory}{kwargs}: mj={mj_r} nt={nt_r}"

    mj.cleanup()
    nt.destroy()


def test_tilted_base_orientation_correct_on_mujoco():
    """A roll of 30deg gives base_orientation = -sin(30)**2 = -0.25 (MuJoCo)."""
    sim = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(sim, _roll_quat(30.0), _Z, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    expected = -(math.sin(math.radians(30.0)) ** 2)
    assert make_predicate("base_orientation")(sim) == pytest.approx(expected, abs=1e-4)
    # A tilted-but-stationary base still tracks zero velocity at zero target.
    assert make_predicate("base_velocity", vx=0.0, vy=0.0, wz=0.0)(sim) == pytest.approx(0.0, abs=1e-4)
    sim.cleanup()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_tilted_base_orientation_matches_across_backends():
    """A rolled base gives the same non-zero orientation penalty on both backends."""
    quat, expected = _roll_quat(30.0), -(math.sin(math.radians(30.0)) ** 2)
    mj = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(mj, quat, _Z, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    nt = _build_newton(_FLOATER_MJCF)
    _set_newton(nt, quat, _Z, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    mj_r = make_predicate("base_orientation", weight=2.0)(mj)
    nt_r = make_predicate("base_orientation", weight=2.0)(nt)
    assert mj_r == pytest.approx(2.0 * expected, abs=1e-4)
    assert nt_r == pytest.approx(mj_r, abs=1e-4)
    mj.cleanup()
    nt.destroy()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_reward_terms_degrade_to_zero_on_newton_fixed_base_arm():
    """On a Newton fixed-base arm (no floating base) every term degrades to 0.0.

    Parity with the MuJoCo-only degradation contract: a spec that references a
    base reward term on a fixed-base arm must not crash or invent a value - the
    term returns 0.0 (and the missing base is logged once).
    """
    eng = _build_newton(_FIXED_ARM_MJCF)
    obs = eng.get_observation("floater", skip_images=True)
    assert "base_pos" not in obs and "base_quat" not in obs
    _reset_resolution_warnings()
    assert make_predicate("base_velocity", vx=1.0, vy=0.0, wz=0.0)(eng) == 0.0
    assert make_predicate("base_height", target=0.5)(eng) == 0.0
    assert make_predicate("base_orientation", weight=2.0)(eng) == 0.0
    assert make_predicate("base_lin_vel_z", weight=2.0)(eng) == 0.0
    assert make_predicate("base_ang_vel_xy", weight=2.0)(eng) == 0.0
    eng.destroy()


# ---- uncommanded-base-velocity regularizers (base_lin_vel_z / base_ang_vel_xy) ----
#
# base_lin_vel_z and base_ang_vel_xy landed after the cross-backend pin above and
# complete the minimal legged_gym reward set by damping the base's UNCOMMANDED
# velocity. They lean on the SAME frame contract as base_velocity - base_lin_vel
# is WORLD (rotated into the base frame via base_quat for the vertical velocity)
# and base_ang_vel is BODY (the roll/pitch RATES read directly) - but on axes the
# moving-base state above leaves at zero (that base glides horizontally and spins
# about +z, so its vertical velocity and roll/pitch rate are both 0). These
# dedicated states drive those axes non-zero on BOTH backends and pin that the
# same physical base state yields the same reward on each.

# base_lin_vel_z: a base ROLLED 30deg about +x moving straight UP in the world at
# 0.7 m/s. World +z is NOT the base's own up-axis once rolled, so the term must
# rotate world->body via base_quat: v_body_z = 0.7 * cos(30) and base_lin_vel_z =
# -(0.7 * cos30)^2. A term that read the raw WORLD vertical velocity would give
# -0.49 instead; the gap proves the body-frame rotation on each backend.
_LIN_ROLL_DEG = 30.0
_LIN_VZ = 0.7
_EXP_V_BODY_Z = _LIN_VZ * math.cos(math.radians(_LIN_ROLL_DEG))
_EXP_LIN_VEL_Z = -(_EXP_V_BODY_Z**2)

# base_ang_vel_xy: a LEVEL base (identity orientation, so world == body, which lets
# the existing world-frame Newton qd writer set a body-frame rate verbatim)
# tipping at wx=0.3 (roll) / wy=-0.5 (pitch) while ALSO spinning fast about yaw
# (wz=2.0). base_ang_vel_xy = -(0.3^2 + 0.5^2) = -0.34 and MUST ignore the 2.0
# yaw rate (a walking policy may turn freely).
_ANG_IDENT = [1.0, 0.0, 0.0, 0.0]
_ANG_ROLL_RATE, _ANG_PITCH_RATE, _ANG_YAW_RATE = 0.3, -0.5, 2.0
_EXP_ANG_VEL_XY = -(_ANG_ROLL_RATE**2 + _ANG_PITCH_RATE**2)


def test_base_lin_vel_z_body_frame_correct_on_mujoco():
    """base_lin_vel_z rotates the world vertical velocity into the base frame (MuJoCo)."""
    sim = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(sim, _roll_quat(_LIN_ROLL_DEG), _Z, [0.0, 0.0, _LIN_VZ], [0.0, 0.0, 0.0])
    assert make_predicate("base_lin_vel_z")(sim) == pytest.approx(_EXP_LIN_VEL_Z, abs=1e-4)
    # Decisive: the body-frame value is NOT the naive raw-world-z penalty.
    assert make_predicate("base_lin_vel_z")(sim) != pytest.approx(-(_LIN_VZ**2), abs=1e-3)
    # No roll/pitch RATE in this state -> the wobble term reads zero.
    assert make_predicate("base_ang_vel_xy")(sim) == pytest.approx(0.0, abs=1e-4)
    sim.cleanup()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_base_lin_vel_z_body_frame_correct_on_newton():
    """The SAME rolled-and-rising base yields the SAME base_lin_vel_z through Newton."""
    eng = _build_newton(_FLOATER_MJCF)
    _set_newton(eng, _roll_quat(_LIN_ROLL_DEG), _Z, [0.0, 0.0, _LIN_VZ], [0.0, 0.0, 0.0])
    assert make_predicate("base_lin_vel_z")(eng) == pytest.approx(_EXP_LIN_VEL_Z, abs=1e-4)
    assert make_predicate("base_ang_vel_xy")(eng) == pytest.approx(0.0, abs=1e-4)
    eng.destroy()


def test_base_ang_vel_xy_yaw_invariant_correct_on_mujoco():
    """base_ang_vel_xy penalises the roll/pitch RATE and ignores the yaw rate (MuJoCo)."""
    sim = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(sim, _ANG_IDENT, _Z, [0.0, 0.0, 0.0], [_ANG_ROLL_RATE, _ANG_PITCH_RATE, _ANG_YAW_RATE])
    assert make_predicate("base_ang_vel_xy")(sim) == pytest.approx(_EXP_ANG_VEL_XY, abs=1e-4)
    # No vertical velocity in this state -> the bounce term reads zero.
    assert make_predicate("base_lin_vel_z")(sim) == pytest.approx(0.0, abs=1e-4)
    sim.cleanup()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_base_ang_vel_xy_yaw_invariant_correct_on_newton():
    """The SAME roll/pitch/yaw rate yields the SAME base_ang_vel_xy through Newton."""
    eng = _build_newton(_FLOATER_MJCF)
    _set_newton(eng, _ANG_IDENT, _Z, [0.0, 0.0, 0.0], [_ANG_ROLL_RATE, _ANG_PITCH_RATE, _ANG_YAW_RATE])
    assert make_predicate("base_ang_vel_xy")(eng) == pytest.approx(_EXP_ANG_VEL_XY, abs=1e-4)
    assert make_predicate("base_lin_vel_z")(eng) == pytest.approx(0.0, abs=1e-4)
    eng.destroy()


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")
def test_uncommanded_velocity_terms_identical_across_backends():
    """MuJoCo == Newton for base_lin_vel_z / base_ang_vel_xy on the same base state.

    The guard the per-backend tests cannot give: the two uncommanded-velocity
    regularizers, fed the SAME physical base state, must return the SAME value on
    both engines. A frame flip or a dropped base key on either backend (the class
    the base linear/angular velocity surfacing was hardened against) makes them
    diverge here even while the per-backend value tests still pass.
    """
    # base_lin_vel_z on the rolled-and-rising base.
    mj = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(mj, _roll_quat(_LIN_ROLL_DEG), _Z, [0.0, 0.0, _LIN_VZ], [0.0, 0.0, 0.0])
    nt = _build_newton(_FLOATER_MJCF)
    _set_newton(nt, _roll_quat(_LIN_ROLL_DEG), _Z, [0.0, 0.0, _LIN_VZ], [0.0, 0.0, 0.0])
    mj_lz = make_predicate("base_lin_vel_z", weight=2.0)(mj)
    nt_lz = make_predicate("base_lin_vel_z", weight=2.0)(nt)
    assert mj_lz == pytest.approx(2.0 * _EXP_LIN_VEL_Z, abs=1e-4)
    assert nt_lz == pytest.approx(mj_lz, abs=1e-4)
    mj.cleanup()
    nt.destroy()

    # base_ang_vel_xy on the wobbling-and-yawing base.
    mj = _build_mujoco(_FLOATER_MJCF)
    _set_mujoco(mj, _ANG_IDENT, _Z, [0.0, 0.0, 0.0], [_ANG_ROLL_RATE, _ANG_PITCH_RATE, _ANG_YAW_RATE])
    nt = _build_newton(_FLOATER_MJCF)
    _set_newton(nt, _ANG_IDENT, _Z, [0.0, 0.0, 0.0], [_ANG_ROLL_RATE, _ANG_PITCH_RATE, _ANG_YAW_RATE])
    mj_axy = make_predicate("base_ang_vel_xy", weight=2.0)(mj)
    nt_axy = make_predicate("base_ang_vel_xy", weight=2.0)(nt)
    assert mj_axy == pytest.approx(2.0 * _EXP_ANG_VEL_XY, abs=1e-4)
    assert nt_axy == pytest.approx(mj_axy, abs=1e-4)
    mj.cleanup()
    nt.destroy()
