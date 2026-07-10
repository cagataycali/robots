"""Regression tests for the ``base_below_z`` floating-base height-collapse predicate.

The predicate/reward DSL reads a floating base's height (``base_pos``) in the
``base_height`` reward term and its orientation (``base_quat``) in
``base_orientation`` / ``base_tipped``, all off the embodiment-agnostic surface
``get_observation`` exposes for a floating base (a base body that may sit on an
UNNAMED free joint). ``base_tipped`` (#1230) added the *topple* half of a
locomotion fall termination on that surface; the *collapse* half - "the torso
dropped to the floor, end the episode" - was still inexpressible without a base
body name. ``body_below_z`` needs a per-embodiment base body name and cannot
reach a mobile base whose free joint is unnamed, and ``base_height`` is a
float-valued reward term (wrong kind for a failure clause).

``base_below_z(z, robot)`` closes that gap: TRUE when the base's world height
has dropped below ``z``, reading the same ``base_pos`` signal ``base_height``
reads, so it drops straight into a ``failure`` clause next to ``base_tipped``.
These tests set a KNOWN base height directly on the sim and assert the
threshold, orientation/yaw independence (it is a pure height predicate, unlike
``base_tipped``), live tracking, fixed-base degradation, and that a real
``DeclarativeBenchmark`` failure clause built from ``base_tipped`` +
``base_below_z`` terminates the episode when the base either topples OR
collapses. They are GL-free (``get_observation`` with ``skip_images``) so they
run in CI without a display.
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

# Fixed-base arm: no free joint anywhere -> no base position. base_below_z must
# degrade to False (and warn) rather than crash or invent a value.
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
    s = Simulation(tool_name="test_base_below_z", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, z: float, quat_wxyz: list[float] | None = None) -> None:
    """Set the robot's (only) free joint to world height ``z`` with orientation quat."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    q = quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, z, *q]
    mujoco.mj_forward(model, data)


def test_base_below_z_is_registered_as_a_bool_predicate():
    """It must classify as bool so the DSL accepts it in success/failure clauses."""
    assert "base_below_z" in PREDICATE_REGISTRY
    assert predicate_kind("base_below_z") == "bool"


def test_base_below_z_trips_when_base_drops_below_the_threshold(sim):
    """FALSE while the base height is at/above z, TRUE once it drops strictly below."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_below_z", z=0.5)
    _set_base_pose(sim, 0.8)
    assert pred(sim) is False
    _set_base_pose(sim, 0.5)  # exactly at the threshold is not "below"
    assert pred(sim) is False
    _set_base_pose(sim, 0.49)
    assert pred(sim) is True
    _set_base_pose(sim, 0.05)  # fully collapsed to the floor
    assert pred(sim) is True


def test_base_below_z_is_independent_of_orientation(sim):
    """It reads only base HEIGHT: a tipped or yawed base at the same height reads
    identically (the pure-height complement of base_tipped, which reads only
    orientation)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_below_z", z=0.5)
    for quat in ([1.0, 0.0, 0.0, 0.0], _axis_quat("x", 80.0), _axis_quat("z", 120.0)):
        _set_base_pose(sim, 0.8, quat)
        assert pred(sim) is False, "upright/tipped/yawed but high -> not below"
        _set_base_pose(sim, 0.2, quat)
        assert pred(sim) is True, "upright/tipped/yawed but low -> below"


def test_base_below_z_tracks_the_live_base_height(sim):
    """The predicate reads the CURRENT base height: lowering the base flips it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_below_z", z=0.4)
    _set_base_pose(sim, 0.74)
    assert pred(sim) is False
    _set_base_pose(sim, 0.1)
    assert pred(sim) is True


def test_base_below_z_accepts_a_below_ground_threshold(sim):
    """z is an unvalidated world height (mirrors body_below_z): a negative z is a
    below-ground threshold that a resting base never trips."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_below_z", z=-0.1)
    _set_base_pose(sim, 0.0)
    assert pred(sim) is False
    _set_base_pose(sim, -0.2)
    assert pred(sim) is True


def test_base_below_z_degrades_to_false_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base position: the predicate degrades to False
    (never collapsed -> never spuriously fails an episode) and warns once."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_below_z", z=0.3)(sim)
    assert val is False
    assert any("base" in r.message.lower() for r in caplog.records)


def test_declarative_benchmark_terminates_on_topple_or_collapse(sim):
    """End to end: a DeclarativeBenchmark failure clause combining base_tipped and
    base_below_z compiles and terminates the episode when the base EITHER topples
    OR collapses - the complete locomotion fall termination a velocity-tracking
    spec needs (neither half alone catches both failure modes)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "walk-forward",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "dense_reward": [{"predicate": "base_velocity", "vx": 1.0, "weight": 1.0}],
            "failure": {
                "any": [
                    {"predicate": "base_tipped", "tol": 0.7},
                    {"predicate": "base_below_z", "z": 0.3},
                ]
            },
        }
    )
    # Upright and standing: episode continues.
    _set_base_pose(sim, 0.8, [1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    # Collapsed to the floor but NOT tipped: base_below_z alone must fire.
    _set_base_pose(sim, 0.1, [1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is True
    # Toppled on its side but still high (0.8): base_tipped alone must fire.
    _set_base_pose(sim, 0.8, _axis_quat("y", 90.0))
    assert bench.is_failure(sim) is True
