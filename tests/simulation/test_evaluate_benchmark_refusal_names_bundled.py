"""``evaluate_benchmark`` on an unregistered name points at the bundled set.

With an empty registry the refusal listed ``Registered: []`` and sent the
caller to ``register_benchmark_from_file`` - for names such as
``go2_walk_forward`` that ship in the box and are one
``register_builtin_benchmarks`` call away (which ``list_benchmarks`` already
says). The refusal now names that call for a bundled name, and the bundled set
for any other name.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import benchmark as benchmark_registry
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


@pytest.fixture
def sim(monkeypatch):
    # An empty registry, whatever earlier tests registered.
    monkeypatch.setattr(benchmark_registry, "_BENCHMARK_REGISTRY", {})
    engine = MuJoCoSimEngine(tool_name="bench_refusal", mesh=False)
    engine.create_world()
    assert engine.add_robot(name="so101", data_config="so101")["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup()


def test_bundled_name_names_register_builtin_benchmarks(sim):
    r = sim.evaluate_benchmark("go2_walk_forward")
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith("evaluate_benchmark: no benchmark registered under 'go2_walk_forward'.")
    assert (
        "'go2_walk_forward' is bundled but not registered yet - call action='register_builtin_benchmarks' first" in text
    )
    assert "register_benchmark_from_file" not in text


def test_unknown_name_lists_the_bundled_set(sim):
    r = sim.evaluate_benchmark("libero")
    assert r["status"] == "error"
    text = _text(r)
    assert (
        "Bundled (register_builtin_benchmarks adds them): ['g1_walk_forward', 'go2_strafe_left', 'go2_turn_left', 'go2_walk_forward', 't1_walk_forward']."
        in text
    )
    assert text.endswith("register_benchmark_from_file or register_benchmark adds your own.")
