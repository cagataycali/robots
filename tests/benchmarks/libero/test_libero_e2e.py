"""End-to-end dispatch test for LiberoAdapter via the MuJoCo Simulation class.

Exercises the full register_benchmark → evaluate_benchmark path through
``_dispatch_action`` with a real MuJoCo world. Does not require the
``libero`` pip package - uses a hand-written BDDL string and inline MJCF.

Distinct from ``tests/simulation/mujoco/test_benchmark_dispatch.py`` which
covers the generic benchmark dispatch path; this test pins the LIBERO
adapter + BDDL compile pipeline against a live sim.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.benchmarks.libero import LiberoAdapter  # noqa: E402
from strands_robots.simulation.benchmark import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    register_benchmark,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Simple Panda-like MJCF - good enough for the benchmark compat check
# (declares data_config=panda on the robot) and for get_body_state lookups.
PANDA_LIKE_XML = """
<mujoco model="panda_lite">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


@pytest.fixture
def sim():
    s = Simulation(tool_name="libero_sim", mesh=False)
    yield s
    s.cleanup()


@pytest.fixture
def robot_xml_path():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "panda_lite.xml")
    with open(path, "w") as f:
        f.write(PANDA_LIKE_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_panda(sim, robot_xml_path):
    sim.create_world()
    # data_config="panda" so LiberoAdapter's compat check passes.
    result = sim.add_robot("panda_arm", urdf_path=robot_xml_path, data_config="panda")
    assert result["status"] == "success"
    return sim


class TestLiberoEvaluateBenchmarkEndToEnd:
    def test_registered_adapter_round_trips_via_dispatcher(self, sim_with_panda):
        """The MuJoCo dispatcher resolves evaluate_benchmark → PolicyRunner →
        LiberoAdapter.is_success without crashing."""
        adapter = LiberoAdapter.from_text(
            """
            (define (problem libero_keep_arm_intact)
              (:language "do anything; success is never")
              (:goal (on nonexistent_cube nonexistent_plate)))
            """,
            max_steps=3,
            init_jitter=0.0,  # nonexistent bodies - don't jitter
        )
        register_benchmark("libero-e2e", adapter)

        result = sim_with_panda._dispatch_action(
            "evaluate_benchmark",
            {
                "action": "evaluate_benchmark",
                "benchmark_name": "libero-e2e",
                "robot_name": "panda_arm",
                "policy_provider": "mock",
                "n_episodes": 2,
                "seed": 7,
            },
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_episodes"] == 2
        assert payload["benchmark_class"] == "LiberoAdapter"
        # Mock policy can't satisfy the goal; success_rate is 0 across both eps.
        assert payload["n_success"] == 0
        # But the loop ran - each episode ran max_steps=3.
        assert all(ep["steps"] == 3 for ep in payload["episodes"])

    def test_non_panda_robot_surfaces_structured_compat_error(self, sim, robot_xml_path):
        """A sim loaded with a non-Panda data_config must produce a structured
        error, not a raw traceback, when the LIBERO adapter evaluates."""
        sim.create_world()
        sim.add_robot("so100_arm", urdf_path=robot_xml_path, data_config="so100")

        adapter = LiberoAdapter.from_text("(define (problem t) (:goal (grasped cube)))")
        register_benchmark("libero-compat-test", adapter)

        result = sim._dispatch_action(
            "evaluate_benchmark",
            {
                "action": "evaluate_benchmark",
                "benchmark_name": "libero-compat-test",
                "robot_name": "so100_arm",
                "policy_provider": "mock",
                "n_episodes": 1,
            },
        )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "compatibility" in text.lower() or "supported" in text.lower()
        assert "so100" in text
        assert "panda" in text
