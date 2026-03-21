"""pytest-benchmark tests for strands_robots.

Run:
    pytest benchmarks/test_bench.py --benchmark-only
    pytest benchmarks/test_bench.py --benchmark-only --benchmark-json=benchmarks/results/pytest_bench.json
    pytest benchmarks/test_bench.py --benchmark-only --benchmark-compare
    pytest benchmarks/test_bench.py --benchmark-only --benchmark-histogram=benchmarks/results/hist
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_policy():
    from strands_robots.policies.mock import MockPolicy

    return MockPolicy(action_space={f"j{i}": 0.0 for i in range(6)})


@pytest.fixture(scope="module")
def mock_obs():
    return {"observation.state": np.zeros(6, dtype=np.float32)}


# ---------------------------------------------------------------------------
# Import benchmarks
# ---------------------------------------------------------------------------


def test_import_strands_robots(benchmark):
    """Benchmark: import strands_robots"""
    import importlib
    import strands_robots

    def reimport():
        importlib.reload(strands_robots)

    benchmark(reimport)


def test_import_registry(benchmark):
    """Benchmark: import registry functions"""

    def do():
        from strands_robots.registry import list_robots, list_aliases  # noqa: F811

        return list_robots(), list_aliases()

    benchmark(do)


def test_import_groot_client(benchmark):
    """Benchmark: import groot client (zmq + msgpack)"""

    def do():
        from strands_robots.policies.groot.client import (  # noqa: F811
            Gr00tInferenceClient,
        )

        return Gr00tInferenceClient

    benchmark(do)


# ---------------------------------------------------------------------------
# Registry benchmarks
# ---------------------------------------------------------------------------


def test_list_robots(benchmark):
    from strands_robots.registry import list_robots

    result = benchmark(list_robots)
    assert len(result) > 0


def test_list_providers(benchmark):
    from strands_robots.policies.factory import list_providers

    result = benchmark(list_providers)
    assert "mock" in result


def test_list_aliases(benchmark):
    from strands_robots.registry import list_aliases

    result = benchmark(list_aliases)
    assert len(result) > 0


def test_get_robot(benchmark):
    from strands_robots.registry import get_robot

    result = benchmark(get_robot, "so100")
    assert result is not None


def test_resolve_policy_groot(benchmark):
    from strands_robots.registry import resolve_policy

    result = benchmark(resolve_policy, "groot")
    assert result[0] == "groot"


def test_format_robot_table(benchmark):
    from strands_robots.registry import format_robot_table

    result = benchmark(format_robot_table)
    assert len(result) > 100


def test_load_data_config(benchmark):
    from strands_robots.policies.groot.data_config import load_data_config

    result = benchmark(load_data_config, "so100")
    assert result.name == "so100"


# ---------------------------------------------------------------------------
# Mock policy benchmarks
# ---------------------------------------------------------------------------


def test_mock_create(benchmark):
    from strands_robots.policies.mock import MockPolicy

    benchmark(MockPolicy, action_space={f"j{i}": 0.0 for i in range(6)})


def test_mock_inference(benchmark, mock_policy, mock_obs):
    """Benchmark: mock policy inference (hot path)"""
    result = benchmark(mock_policy.get_actions_sync, mock_obs, "bench")
    assert len(result) > 0
