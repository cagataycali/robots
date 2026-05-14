"""Dispatch-path tests for the benchmark tool actions on the MuJoCo backend.

Mirrors the ``test_agenttool_contract.py`` pattern: exercises ``_dispatch_action``
with the new action names (``list_benchmarks``, ``register_benchmark_from_file``,
``evaluate_benchmark``) and asserts:

* the tool_spec ``action`` enum exposes them,
* unknown / missing params produce the friendly structured errors that the
  dispatcher generates from ``inspect.signature``,
* the underlying ``SimEngine`` facade is reached for valid inputs.

Integration with real MuJoCo physics is deferred to ``tests_integ/``; these
tests only need the Simulation stub without a created world.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import _BENCHMARK_REGISTRY  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Robot XML reused from test_simulation.py (keep in sync if the canonical
# fixture changes).
ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
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
    s = Simulation(tool_name="bench_sim", mesh=False)
    yield s
    s.cleanup()


@pytest.fixture
def robot_xml_path():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_robot(sim, robot_xml_path):
    sim.create_world()
    sim.add_robot("arm1", urdf_path=robot_xml_path)
    return sim


@pytest.fixture
def basic_spec_file(tmp_path: Path):
    path = tmp_path / "basic.json"
    path.write_text(
        json.dumps(
            {
                "name": "basic-task",
                "default_robot": "arm1",
                "supported_robots": [],  # any robot (so100 by data_config isn't loaded here)
                "max_steps": 5,
            }
        )
    )
    return str(path)


# Tool spec: action enum + property surface


class TestToolSpecSurface:
    def test_enum_includes_new_actions(self, sim):
        # _TOOL_SPEC_SCHEMA lives at module level; read via module introspection.
        from strands_robots.simulation.mujoco import simulation as _sim_mod

        enum = _sim_mod._TOOL_SPEC_SCHEMA["properties"]["action"]["enum"]
        assert "list_benchmarks" in enum
        assert "register_benchmark_from_file" in enum
        assert "evaluate_benchmark" in enum

    def test_property_surface_has_new_params(self):
        from strands_robots.simulation.mujoco import simulation as _sim_mod

        props = _sim_mod._TOOL_SPEC_SCHEMA["properties"]
        assert "benchmark_name" in props
        assert "spec_path" in props


# Dispatch


class TestListBenchmarksDispatch:
    def test_empty_registry(self, sim):
        result = sim._dispatch_action("list_benchmarks", {"action": "list_benchmarks"})
        assert result["status"] == "success"
        assert "No benchmarks" in result["content"][0]["text"]

    def test_lists_registered(self, sim, basic_spec_file):
        sim._dispatch_action(
            "register_benchmark_from_file",
            {
                "action": "register_benchmark_from_file",
                "benchmark_name": "basic",
                "spec_path": basic_spec_file,
            },
        )
        result = sim._dispatch_action("list_benchmarks", {"action": "list_benchmarks"})
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert "basic" in payload["benchmarks"]


class TestRegisterBenchmarkFromFileDispatch:
    def test_happy_path(self, sim, basic_spec_file):
        result = sim._dispatch_action(
            "register_benchmark_from_file",
            {
                "action": "register_benchmark_from_file",
                "benchmark_name": "happy",
                "spec_path": basic_spec_file,
            },
        )
        assert result["status"] == "success"
        assert "happy" in result["content"][0]["text"]

    def test_no_args_friendly_error(self, sim):
        """Dispatcher surfaces missing required params with a clear message."""
        result = sim._dispatch_action("register_benchmark_from_file", {"action": "register_benchmark_from_file"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "requires parameter" in text

    def test_bad_spec_path_returns_structured_error(self, sim):
        result = sim._dispatch_action(
            "register_benchmark_from_file",
            {
                "action": "register_benchmark_from_file",
                "benchmark_name": "missing",
                "spec_path": "/nonexistent/path/nope.json",
            },
        )
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"].lower()

    def test_unknown_param_rejected(self, sim, basic_spec_file):
        """Dispatcher rejects unknown kwargs so spec drift is caught early."""
        result = sim._dispatch_action(
            "register_benchmark_from_file",
            {
                "action": "register_benchmark_from_file",
                "benchmark_name": "x",
                "spec_path": basic_spec_file,
                "bogus": 1,
            },
        )
        assert result["status"] == "error"
        assert "Unknown parameter 'bogus'" in result["content"][0]["text"]


class TestEvaluateBenchmarkDispatch:
    def test_requires_benchmark_name(self, sim):
        result = sim._dispatch_action("evaluate_benchmark", {"action": "evaluate_benchmark"})
        assert result["status"] == "error"
        assert "requires parameter" in result["content"][0]["text"]

    def test_unknown_benchmark(self, sim_with_robot):
        result = sim_with_robot._dispatch_action(
            "evaluate_benchmark",
            {"action": "evaluate_benchmark", "benchmark_name": "never-registered"},
        )
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_evaluate_end_to_end(self, sim_with_robot, basic_spec_file):
        """Register a no-op benchmark, evaluate with the mock policy - must succeed."""
        sim_with_robot._dispatch_action(
            "register_benchmark_from_file",
            {
                "action": "register_benchmark_from_file",
                "benchmark_name": "e2e",
                "spec_path": basic_spec_file,
            },
        )
        result = sim_with_robot._dispatch_action(
            "evaluate_benchmark",
            {
                "action": "evaluate_benchmark",
                "benchmark_name": "e2e",
                "robot_name": "arm1",
                "policy_provider": "mock",
                "n_episodes": 1,
                "seed": 0,
            },
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_episodes"] == 1
        assert payload["benchmark_class"] == "DeclarativeBenchmark"
        # With no success/failure/done in the spec, loop runs to max_steps=5.
        assert payload["episodes"][0]["steps"] == 5
