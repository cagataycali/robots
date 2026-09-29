"""``evaluate_benchmark`` carries the inference-cost pair ``eval_policy`` carries.

docs/learn/simulation/predicates-and-rollouts.md promises that the json block of
``eval_policy`` AND ``evaluate_benchmark`` carries inference timing
(``avg_inference_ms``, ``max_inference_ms``, RTC counters). The spec path never
collected it (#4146): the same policy, robot and horizon reported the pair from
``eval_policy`` and nothing from ``evaluate_benchmark``.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

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

INFERENCE_KEYS = (
    "avg_inference_ms",
    "max_inference_ms",
    "chunk_prefetch_enabled",
    "chunk_prefetch_chunks_acquired",
    "chunk_prefetch_hits",
    "chunk_prefetch_blocks",
    "policy_rtc_enabled",
)


class _FullHorizon(BenchmarkProtocol):
    max_steps = 6

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "arm1"

    def on_step(self, sim, obs, action) -> StepInfo:
        return StepInfo(reward=0.0)

    def is_success(self, sim) -> bool:
        return False

    def is_failure(self, sim) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


@pytest.fixture
def sim_with_robot():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    s = Simulation(tool_name="bench_inference_sim", mesh=False)
    s.create_world()
    s.add_robot("arm1", urdf_path=path)
    yield s
    s.cleanup()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


def test_evaluate_benchmark_reports_the_same_inference_keys_as_eval_policy(sim_with_robot):
    register_benchmark("full_horizon", _FullHorizon())
    bench = _json(
        sim_with_robot.evaluate_benchmark(
            "full_horizon", robot_name="arm1", policy_provider="mock", n_episodes=2, action_horizon=2
        )
    )
    ev = _json(
        sim_with_robot.eval_policy(
            robot_name="arm1", policy_provider="mock", n_episodes=2, max_steps=6, action_horizon=2
        )
    )
    missing = [k for k in INFERENCE_KEYS if k not in bench]
    assert not missing, (
        f"evaluate_benchmark json lacks {missing}; eval_policy carries {[k for k in INFERENCE_KEYS if k in ev]}"
    )
    # Same policy, robot, horizon and episode count: the two surfaces query the policy the
    # same number of times, 3 chunks of 2 per 6-step episode. The spec loop used to run one
    # iteration per step and query on each, so it paid 12 inferences for the 6 it used.
    assert bench["chunk_prefetch_chunks_acquired"] == ev["chunk_prefetch_chunks_acquired"] == 6
    assert bench["chunk_prefetch_enabled"] is False
    assert bench["chunk_prefetch_hits"] == 0 and bench["chunk_prefetch_blocks"] == 0
    assert bench["avg_inference_ms"] >= 0.0
    assert bench["max_inference_ms"] >= bench["avg_inference_ms"]
    # The pre-rename spellings stay for one release, as on the eval_policy path.
    assert bench["rtc_avg_inference_ms"] == bench["avg_inference_ms"]
    assert bench["rtc_max_inference_ms"] == bench["max_inference_ms"]
