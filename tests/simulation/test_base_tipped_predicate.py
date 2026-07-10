"""Regression tests for the ``base_tipped`` locomotion fall-over predicate.

The predicate/reward DSL grew all five legged_gym reward terms
(``base_velocity`` / ``base_height`` / ``base_orientation`` / ``base_lin_vel_z``
/ ``base_ang_vel_xy``) reading the embodiment-agnostic floating-base surface
from ``get_observation``, but it had no BOOL predicate on that surface. So a
locomotion benchmark's canonical *termination* - "the robot fell over, end the
episode" - was inexpressible: the DSL has no ``not`` operator to negate
``body_upright``, and ``body_upright`` in a ``success`` clause is the wrong
polarity for a survival task (it would report instant success on step 0). The
only workaround was ``body_below_z(<base body name>)`` / ``body_upright(<base
body name>)``, which needs a per-embodiment base body name and cannot reach a
mobile base whose free joint is unnamed.

``base_tipped(tol, robot)`` closes that gap. It is TRUE when the floating base
has tilted more than ``tol`` from level - the exact complement of
``body_upright``'s upright check applied to the ``base_quat`` signal - so it
drops straight into a ``failure`` clause. These tests set a KNOWN base
orientation directly on the sim and assert the tilt threshold, yaw invariance,
roll/pitch symmetry, live tracking, fixed-base degradation, ``tol`` validation,
and that a real ``DeclarativeBenchmark`` failure clause built from it terminates
the episode when the base tips. They are GL-free (``get_observation`` with
``skip_images``) so they run in CI without a display.
"""

import logging
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    _reset_resolution_warnings,
    make_predicate,
    predicate_kind,
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

# Fixed-base arm: no free joint anywhere -> no base orientation. base_tipped
# must degrade to False (and warn) rather than crash or invent a value.
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


def _tilt_metric(deg: float) -> float:
    """The predicate's tilt quantity 2*(x^2+y^2) for a roll/pitch of ``deg`` = 1 - cos(deg)."""
    return 1.0 - math.cos(math.radians(deg))


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_tipped", mesh=False)
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


def test_base_tipped_is_registered_as_a_bool_predicate():
    """It must classify as bool so the DSL accepts it in success/failure clauses."""
    assert "base_tipped" in PREDICATE_REGISTRY
    assert predicate_kind("base_tipped") == "bool"


def test_base_tipped_is_false_when_level(sim):
    """A perfectly level base is not tipped at any tol >= 0."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_base_quat(sim, [1.0, 0.0, 0.0, 0.0])
    assert make_predicate("base_tipped")(sim) is False
    assert make_predicate("base_tipped", tol=0.0)(sim) is False


def test_base_tipped_trips_past_the_tol_threshold(sim):
    """TRUE once the tilt quantity 2*(x^2+y^2) reaches tol; FALSE just below it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    # tol=0.5 (1 - cos(theta) > 0.5): 45 deg (0.293) is below, 70 deg (0.658) is above.
    below = make_predicate("base_tipped", tol=0.5)
    _set_base_quat(sim, _axis_quat("x", 45.0))
    assert _tilt_metric(45.0) < 0.5 and below(sim) is False
    _set_base_quat(sim, _axis_quat("x", 70.0))
    assert _tilt_metric(70.0) > 0.5 and below(sim) is True
    # The default tol=0.15 trips at ~32 deg: a 10 deg lean is upright, 45 deg is tipped.
    default = make_predicate("base_tipped")
    _set_base_quat(sim, _axis_quat("x", 10.0))
    assert _tilt_metric(10.0) < 0.15 and default(sim) is False
    _set_base_quat(sim, _axis_quat("x", 45.0))
    assert _tilt_metric(45.0) >= 0.15 and default(sim) is True


def test_base_tipped_is_symmetric_roll_and_pitch(sim):
    """A pitch of theta trips identically to a roll of theta (isotropic tilt)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_tipped", tol=0.4)  # 1 - cos(theta) > 0.4 -> theta > ~53 deg
    _set_base_quat(sim, _axis_quat("x", 65.0))
    roll = term(sim)
    _set_base_quat(sim, _axis_quat("y", 65.0))
    pitch = term(sim)
    assert roll is True and pitch is True
    _set_base_quat(sim, _axis_quat("x", 40.0))
    roll_lo = term(sim)
    _set_base_quat(sim, _axis_quat("y", 40.0))
    pitch_lo = term(sim)
    assert roll_lo is False and pitch_lo is False


def test_base_tipped_is_invariant_to_yaw(sim):
    """Pure yaw (heading change) keeps the base level -> never tipped, so a
    turning-in-place walker is not spuriously terminated."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_tipped", tol=0.05)  # tight bound
    for yaw in (45.0, 90.0, 179.0):
        _set_base_quat(sim, _axis_quat("z", yaw))
        assert term(sim) is False, f"yaw {yaw} must not count as tipped"


def test_base_tipped_tracks_the_live_base_orientation(sim):
    """The predicate reads the CURRENT base orientation: tipping the base flips it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    term = make_predicate("base_tipped", tol=0.5)
    _set_base_quat(sim, [1.0, 0.0, 0.0, 0.0])
    assert term(sim) is False
    _set_base_quat(sim, _axis_quat("y", 90.0))  # fully on its side, 2*(x^2+y^2)=1.0
    assert term(sim) is True


def test_base_tipped_degrades_to_false_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base orientation: the predicate degrades to False
    (never tipped -> never spuriously fails an episode) and warns once."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_tipped")(sim)
    assert val is False
    assert any("base" in r.message.lower() for r in caplog.records)


def test_base_tipped_rejects_negative_tol():
    with pytest.raises(ValueError, match="tol"):
        make_predicate("base_tipped", tol=-0.1)


def test_declarative_benchmark_terminates_on_base_tipped(sim):
    """End to end: a DeclarativeBenchmark failure clause built from base_tipped
    compiles (it did not before this predicate existed) and its is_failure()
    fires only once the base tips past tol - the locomotion fall-over
    termination a velocity-tracking spec needs."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "walk-forward",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "dense_reward": [{"predicate": "base_velocity", "vx": 1.0, "weight": 1.0}],
            "failure": {"any": [{"predicate": "base_tipped", "tol": 0.7}]},
        }
    )
    _set_base_quat(sim, [1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False  # upright: episode continues
    _set_base_quat(sim, _axis_quat("y", 90.0))
    assert bench.is_failure(sim) is True  # toppled: episode terminates
