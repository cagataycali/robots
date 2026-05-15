"""Tests for ``PolicyRunner.evaluate`` with a :class:`BenchmarkProtocol` spec.

Covers the new spec-driven evaluation path:

* Cumulative reward accounting across an episode.
* Early termination on ``is_success`` / ``is_failure`` / ``StepInfo.done``.
* Per-episode seeded RNG so identical seeds produce identical rollouts.
* Robot-compatibility error surfaces as a structured error dict (not raises).
* Legacy ``success_fn`` path still works.
* Passing both ``spec`` and ``success_fn`` is an error.
* :meth:`SimEngine.evaluate_benchmark` facade end-to-end.
* :meth:`SimEngine.list_benchmarks` / :meth:`register_benchmark_from_file`
  facades return structured dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.policy_runner import PolicyRunner

# Fixtures


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


class _FakeRobot:
    """Simple object with a data_config attr so the base compat check passes."""

    def __init__(self, data_config: str):
        self.data_config = data_config


class _FakeWorld:
    """Minimal world with a ``robots`` dict - enough for the base compat check."""

    def __init__(self, robots: dict[str, _FakeRobot]):
        self.robots = dict(robots)


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` that records a few per-episode counters.

    Deliberately stripped-down: no cameras, no physics - just enough for
    :class:`PolicyRunner` to step through.
    """

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1"), data_config: str = "so100"):
        self._joint_names = list(joint_names)
        self._data_config = data_config
        self._step_count = 0
        self._reset_count = 0
        self._world = _FakeWorld({"fake_robot": _FakeRobot(data_config)})

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._step_count = 0
        self._reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self._step_count += n_steps
        return {"status": "success"}

    def get_state(self):
        return {"step_count": self._step_count}

    def add_robot(self, name, **kw):
        data_config = kw.get("data_config") or self._data_config
        self._world.robots[name] = _FakeRobot(data_config)
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joint_names)

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self._step_count += 1

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}


# Test benchmarks


class _CountingBenchmark(BenchmarkProtocol):
    """Benchmark that tracks how many times each hook was called and rewards +1/step."""

    max_steps = 20

    def __init__(self, *, success_after: int = 10**9, fail_after: int = 10**9):
        self.success_after = success_after
        self.fail_after = fail_after
        self.on_episode_start_calls = 0
        self.on_step_calls = 0
        self.rng_seeds_seen: list[int] = []

    @property
    def supported_robots(self) -> list[str]:
        return ["so100"]

    @property
    def default_robot(self) -> str:
        return "so100"

    def on_episode_start(self, sim, rng):
        self.on_episode_start_calls += 1
        # Record the first draw for reproducibility tests.
        self.rng_seeds_seen.append(rng.randint(0, 1000000))
        super().on_episode_start(sim, rng)

    def on_step(self, sim, obs, action):
        self.on_step_calls += 1
        return StepInfo(reward=1.0)

    def is_success(self, sim):
        return self.on_step_calls >= self.success_after

    def is_failure(self, sim):
        return self.on_step_calls >= self.fail_after


class _DoneAfterBenchmark(BenchmarkProtocol):
    """Returns StepInfo(done=True) on the Nth step."""

    max_steps = 50

    def __init__(self, done_after: int):
        self._done_after = done_after
        self._step = 0

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "so100"

    def on_step(self, sim, obs, action):
        self._step += 1
        return StepInfo(reward=0.5, done=self._step >= self._done_after)

    def is_success(self, sim):
        return False

    def is_failure(self, sim):
        return False


# Cumulative reward


class TestCumulativeReward:
    def test_sums_reward_across_steps(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            spec=spec,
            n_episodes=1,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # _CountingBenchmark rewards +1/step; max_steps=20 → 20.0 cumulative.
        assert payload["episodes"][0]["cumulative_reward"] == pytest.approx(20.0)
        assert payload["avg_reward"] == pytest.approx(20.0)

    def test_success_terminates_and_stops_reward_accumulation(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark(success_after=5)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 1
        # Reward per step = 1; terminates on the 5th step.
        assert payload["episodes"][0]["cumulative_reward"] == pytest.approx(5.0)
        assert payload["episodes"][0]["steps"] == 5

    def test_failure_marks_episode_unsuccessful(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark(fail_after=3)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_failure"] == 1
        assert payload["n_success"] == 0
        assert payload["episodes"][0]["failure"] is True
        assert payload["episodes"][0]["success"] is False

    def test_done_flag_terminates_episode(self):
        """StepInfo.done=True ends the episode even without is_success/is_failure."""
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _DoneAfterBenchmark(done_after=4)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["episodes"][0]["steps"] == 4


# Seed reproducibility


class TestSeedReproducibility:
    def test_same_seed_same_rng_draws(self):
        """Two evaluations with the same seed must produce identical per-episode RNG draws."""
        sim1 = FakeSim()
        sim2 = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim1.robot_joint_names("fake_robot"))

        spec1 = _CountingBenchmark()
        spec2 = _CountingBenchmark()

        PolicyRunner(sim1).evaluate("fake_robot", policy, spec=spec1, n_episodes=3, seed=42)
        PolicyRunner(sim2).evaluate("fake_robot", policy, spec=spec2, n_episodes=3, seed=42)

        assert spec1.rng_seeds_seen == spec2.rng_seeds_seen

    def test_different_seed_different_rng_draws(self):
        sim1 = FakeSim()
        sim2 = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim1.robot_joint_names("fake_robot"))

        spec1 = _CountingBenchmark()
        spec2 = _CountingBenchmark()

        PolicyRunner(sim1).evaluate("fake_robot", policy, spec=spec1, n_episodes=3, seed=42)
        PolicyRunner(sim2).evaluate("fake_robot", policy, spec=spec2, n_episodes=3, seed=999)

        assert spec1.rng_seeds_seen != spec2.rng_seeds_seen

    def test_seed_recorded_in_per_episode_results(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_CountingBenchmark(), n_episodes=3, seed=7)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # Every episode has a distinct seed derived from the master seed.
        seeds = [e["seed"] for e in payload["episodes"]]
        assert len(set(seeds)) == 3
        assert payload["seed"] == 7


# Robot compatibility via the spec path


class TestRobotCompatibility:
    def test_mismatched_robot_returns_structured_error(self):
        """Spec with supported=['panda'] vs sim loaded with 'so100' → structured error."""

        class _PandaOnly(_CountingBenchmark):
            @property
            def supported_robots(self) -> list[str]:
                return ["panda"]

            @property
            def default_robot(self) -> str:
                return "panda"

        sim = FakeSim(data_config="so100")  # loaded with so100
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_PandaOnly(), n_episodes=1)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "compatibility" in text.lower() or "supported" in text.lower()
        assert "so100" in text  # shows the offending data_config
        assert "panda" in text  # shows the allowed list


# Legacy success_fn path still works


class TestBackwardCompatibility:
    def test_legacy_success_fn_callable_still_works(self):
        """The pre-PR success_fn=callable path must be unchanged."""
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            n_episodes=2,
            max_steps=5,
            success_fn=lambda _obs: True,
        )
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["success_rate"] == 1.0
        # Legacy path doesn't emit cumulative_reward; just the pre-PR schema.
        assert "cumulative_reward" not in payload

    def test_legacy_success_fn_none_returns_zero_success(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, n_episodes=1, max_steps=3, success_fn=None)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 0

    def test_cannot_pass_both_spec_and_success_fn(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            spec=_CountingBenchmark(),
            success_fn=lambda _obs: True,
        )
        assert result["status"] == "error"
        assert "both" in result["content"][0]["text"].lower()


# SimEngine facade


class TestSimEngineFacades:
    def test_evaluate_benchmark_dispatches_to_runner(self):
        sim = FakeSim()
        spec = _CountingBenchmark()
        register_benchmark("eval-test", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="eval-test",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["benchmark_class"] == "_CountingBenchmark"
        # Full per-step traversal (nothing terminates early).
        assert payload["episodes"][0]["steps"] == 20

    def test_evaluate_benchmark_unknown_name(self):
        sim = FakeSim()
        result = sim.evaluate_benchmark(benchmark_name="never-registered")
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_evaluate_benchmark_auto_picks_sole_robot(self):
        """Single-robot scene: robot_name can be omitted."""
        sim = FakeSim()
        register_benchmark("auto-robot", _CountingBenchmark())
        result = sim.evaluate_benchmark(benchmark_name="auto-robot", n_episodes=1)
        assert result["status"] == "success"

    def test_evaluate_benchmark_requires_robot_name_in_multi_robot(self):
        sim = FakeSim()
        sim.add_robot("second", data_config="so100")
        register_benchmark("multi", _CountingBenchmark())
        result = sim.evaluate_benchmark(benchmark_name="multi")
        assert result["status"] == "error"
        assert "robot_name" in result["content"][0]["text"]

    def test_list_benchmarks_returns_snapshot(self):
        sim = FakeSim()
        register_benchmark("a", _CountingBenchmark())
        register_benchmark("b", _CountingBenchmark())
        result = sim.list_benchmarks()
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert set(payload["benchmarks"].keys()) == {"a", "b"}

    def test_list_benchmarks_empty(self):
        sim = FakeSim()
        result = sim.list_benchmarks()
        assert result["status"] == "success"
        assert "No benchmarks" in result["content"][0]["text"]

    def test_register_benchmark_from_file_success(self, tmp_path: Path):
        sim = FakeSim()
        spec_path = tmp_path / "s.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "file-bench",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                    "max_steps": 7,
                }
            )
        )
        result = sim.register_benchmark_from_file(benchmark_name="file-bench", spec_path=str(spec_path))
        assert result["status"] == "success"
        assert "Registered benchmark" in result["content"][0]["text"]

    def test_register_benchmark_from_file_missing_file(self, tmp_path: Path):
        sim = FakeSim()
        result = sim.register_benchmark_from_file(benchmark_name="missing", spec_path=str(tmp_path / "nope.json"))
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"].lower()

    def test_register_benchmark_from_file_empty_name(self):
        sim = FakeSim()
        result = sim.register_benchmark_from_file(benchmark_name="", spec_path="/tmp/x.json")
        assert result["status"] == "error"
        assert "benchmark_name" in result["content"][0]["text"]

    def test_register_benchmark_from_file_bad_schema(self, tmp_path: Path):
        sim = FakeSim()
        spec_path = tmp_path / "bad.json"
        spec_path.write_text('{"name": "x"}')  # missing default_robot
        result = sim.register_benchmark_from_file(benchmark_name="bad", spec_path=str(spec_path))
        assert result["status"] == "error"
        assert "default_robot" in result["content"][0]["text"]
