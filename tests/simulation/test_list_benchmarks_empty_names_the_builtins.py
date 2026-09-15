"""An empty list_benchmarks names both ways to get one.

The bundled benchmarks are not registered until ``register_builtin_benchmarks``
is called, and the empty-registry text used to name only
``register_benchmark_from_file``. An agent asked for a baseline had to find
the built-in registration action by scanning the action enum.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import benchmark as _benchmark


class _Engine:
    from strands_robots.simulation.base import SimEngine as _Base

    list_benchmarks = _Base.list_benchmarks


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(_benchmark, "list_benchmarks", lambda: {})


def test_empty_text_names_both_registration_actions(empty_registry):
    result = _Engine().list_benchmarks()
    text = result["content"][0]["text"]
    assert result["status"] == "success"
    assert "No benchmarks registered" in text
    assert "register_builtin_benchmarks" in text
    assert "register_benchmark_from_file" in text
    assert result["content"][1]["json"] == {"benchmarks": {}}


def test_empty_text_lists_every_bundled_name(empty_registry):
    from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs

    text = _Engine().list_benchmarks()["content"][0]["text"]
    for name in builtin_benchmark_specs():
        assert name in text, f"bundled benchmark {name!r} is not named in the empty-registry text"
