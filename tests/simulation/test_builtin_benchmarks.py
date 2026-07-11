"""Regression tests for the shipped built-in benchmark specs.

The floating-base predicate/reward DSL (``base_velocity_tracking`` /
``base_height`` / ``base_orientation`` reward terms and the ``base_beyond_x`` /
``base_tipped`` / ``base_below_z`` predicates) was complete but wired into no
runnable benchmark: :func:`list_benchmarks` was empty until a caller
hand-authored a spec. :func:`register_builtin_benchmarks` ships a canonical
velocity-tracking locomotion benchmark (``go2_walk_forward``) composed from
those primitives so a floating-base robot has a discoverable, runnable eval out
of the box.

These tests (1) pin that ``register_builtin_benchmarks`` puts ``go2_walk_forward``
into the registry with the right metadata, (2) drive the compiled
success/failure/dense-reward functions on a REAL inline floating-base MuJoCo sim
at KNOWN base poses to prove the shipped spec's DSL wiring works on physics, and
(3) pin the opt-in / idempotent / non-mutating contract. They are GL-free
(``get_observation`` with ``skip_images``) so they run in CI without a display,
and they use no downloaded robot asset (the compiled predicates are driven
directly, bypassing the ``supported_robots`` robot-load check that only fires in
``on_episode_start``).
"""

import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.benchmark import (
    get_benchmark,
    list_benchmarks,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.builtin_benchmarks import (
    builtin_benchmark_specs,
    register_builtin_benchmarks,
)
from strands_robots.simulation.mujoco.simulation import Simulation

# Floating base with a NAMED free joint plus one actuated hinge (mirrors the
# base_tipped / base_below_z regression fixtures). get_observation surfaces
# base_pos / base_quat / base_lin_vel / base_ang_vel for this robot, which is
# all the go2_walk_forward predicates read.
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


def _quat_pitch(deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a pitch (rotation about world +y)."""
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, math.sin(h), 0.0]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_builtin_benchmarks", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the module-global benchmark registry clean around each test."""
    unregister_benchmark("go2_walk_forward")
    yield
    unregister_benchmark("go2_walk_forward")


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, x: float = 0.0, z: float = 0.8, quat_wxyz=None) -> None:
    """Set the robot's (only) free joint to a known world pose (x, z, orientation)."""
    if quat_wxyz is None:
        quat_wxyz = [1.0, 0.0, 0.0, 0.0]
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [float(x), 0.0, float(z), *quat_wxyz]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Registration + discovery contract
# ---------------------------------------------------------------------------
def test_register_builtin_benchmarks_registers_go2_walk_forward():
    """The shipped go2_walk_forward benchmark is absent until registered, then
    discoverable via list_benchmarks with the expected metadata."""
    assert "go2_walk_forward" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert "go2_walk_forward" in names
    snap = list_benchmarks()
    assert "go2_walk_forward" in snap
    meta = snap["go2_walk_forward"]
    assert meta["class"] == "DeclarativeBenchmark"
    assert meta["supported_robots"] == ["unitree_go2"]
    assert meta["default_robot"] == "unitree_go2"
    assert meta["max_steps"] == 1000


def test_register_builtin_benchmarks_is_idempotent():
    """Re-registering overwrites without raising and keeps a single entry."""
    register_builtin_benchmarks()
    first = get_benchmark("go2_walk_forward")
    register_builtin_benchmarks()
    second = get_benchmark("go2_walk_forward")
    assert isinstance(first, DeclarativeBenchmark)
    assert isinstance(second, DeclarativeBenchmark)
    # a fresh instance each call (compiled from the spec), still one registry key
    assert list(list_benchmarks()).count("go2_walk_forward") == 1


def test_builtin_benchmark_specs_returns_defensive_copies():
    """Callers get deep copies; mutating one cannot corrupt the shipped spec."""
    a = builtin_benchmark_specs()
    assert "go2_walk_forward" in a
    a["go2_walk_forward"]["max_steps"] = -999
    a["go2_walk_forward"]["success"]["all"].clear()
    b = builtin_benchmark_specs()
    assert b["go2_walk_forward"]["max_steps"] == 1000
    assert b["go2_walk_forward"]["success"]["all"]


def test_facade_register_builtin_benchmarks(sim):
    """The SimEngine facade registers the built-ins and reports them."""
    res = sim.register_builtin_benchmarks()
    assert res["status"] == "success"
    payload = next(c["json"] for c in res["content"] if "json" in c)
    assert "go2_walk_forward" in payload["registered"]
    assert "go2_walk_forward" in list_benchmarks()


# ---------------------------------------------------------------------------
# The shipped spec compiles into predicates that work on real physics
# ---------------------------------------------------------------------------
def test_go2_walk_forward_success_fires_on_forward_progress(sim):
    """success = base_beyond_x(2.0): False until the base walks past x=2 m."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0)
    assert bench.is_success(sim) is False
    _set_base_pose(sim, x=1.5)
    assert bench.is_success(sim) is False  # short of the 2 m line
    _set_base_pose(sim, x=3.0)
    assert bench.is_success(sim) is True


def test_go2_walk_forward_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level base continues, a toppled one
    terminates the episode."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_go2_walk_forward_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.18): a standing base continues, a
    collapsed one terminates."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.32)  # nominal Go2 stance
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.1)  # collapsed below 0.18
    assert bench.is_failure(sim) is True


def test_go2_walk_forward_dense_reward_is_finite(sim):
    """on_step sums base_velocity_tracking + base_height + base_orientation into
    a finite dense reward the RL/BC loop can shape on."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.32)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False
