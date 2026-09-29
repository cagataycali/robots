"""A robot added under a registry alias runs the benchmark written for its canonical name.

``add_robot("go2")`` then ``evaluate_benchmark("go2_walk_forward")`` was refused
because the compatibility check compared the alias string against
``supported_robots=["unitree_go2"]`` literally (#4160), while
docs/learn/simulation/index.md lists ``go2`` among the registry names. Both
check sites (the up-front one in ``evaluate_benchmark`` and
``BenchmarkProtocol.on_episode_start``) now resolve each side through the
registry.
"""

from __future__ import annotations

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.registry import get_robot  # noqa: E402
from strands_robots.simulation.benchmark import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkCompatibilityError,
    robot_model_is_supported,
)
from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


def _one_alias_per_builtin() -> list[tuple[str, str, str]]:
    """(benchmark, canonical robot, one registry alias for it) for every builtin with an alias."""
    out = []
    for name, spec in builtin_benchmark_specs().items():
        canonical = spec["default_robot"]
        aliases = (get_robot(canonical) or {}).get("aliases") or []
        if aliases:
            out.append((name, canonical, aliases[0]))
    return out


class TestTheFold:
    def test_an_alias_matches_its_canonical_name(self) -> None:
        assert robot_model_is_supported("go2", ["unitree_go2"])
        assert robot_model_is_supported("unitree_go2", ["go2"])
        assert robot_model_is_supported("g1", ["unitree_g1"])

    def test_an_unrelated_robot_still_does_not(self) -> None:
        assert not robot_model_is_supported("so101", ["unitree_go2"])
        assert not robot_model_is_supported("not_a_robot", ["unitree_go2"])

    def test_the_population_is_not_empty(self) -> None:
        assert len(_one_alias_per_builtin()) >= 3, _one_alias_per_builtin()


@pytest.mark.parametrize(("benchmark", "canonical", "alias"), _one_alias_per_builtin())
def test_a_builtin_benchmark_runs_on_its_robot_added_under_an_alias(benchmark: str, canonical: str, alias: str) -> None:
    sim = Simulation(tool_name="bench_alias_sim", mesh=False)
    try:
        sim.create_world()
        sim.register_builtin_benchmarks()
        sim.add_robot(alias)
        result = sim.evaluate_benchmark(benchmark, robot_name=alias, policy_provider="mock", n_episodes=1, seed=1)
        assert result["status"] == "success", result
        text = next(c["text"] for c in result["content"] if "text" in c)
        assert "is written for" not in text, text
        assert f"on '{alias}'" in text, text
    finally:
        sim.cleanup()


def test_the_up_front_check_still_refuses_a_different_robot() -> None:
    sim = Simulation(tool_name="bench_alias_refusal_sim", mesh=False)
    try:
        sim.create_world()
        sim.register_builtin_benchmarks()
        sim.add_robot("so101")
        result = sim.evaluate_benchmark("go2_walk_forward", robot_name="so101", policy_provider="mock", n_episodes=1)
        assert result["status"] == "error"
        assert "is written for ['unitree_go2']" in next(c["text"] for c in result["content"] if "text" in c)
    finally:
        sim.cleanup()


def test_on_episode_start_agrees_with_the_fold() -> None:
    import random

    from strands_robots.simulation.benchmark import get_benchmark

    sim = Simulation(tool_name="bench_alias_episode_sim", mesh=False)
    try:
        sim.create_world()
        sim.register_builtin_benchmarks()
        sim.add_robot("go2")
        spec = get_benchmark("go2_walk_forward")
        assert spec is not None
        spec.on_episode_start(sim, random.Random(0))  # the alias passes the per-episode check too
        sim.add_robot("so101")
        with pytest.raises(BenchmarkCompatibilityError):
            spec.on_episode_start(sim, random.Random(0))  # a bystander that is not supported is still refused
    finally:
        sim.cleanup()
