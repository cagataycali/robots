"""The bundled benchmark roster is derived, not hand-written.

The bundled benchmarks are not registered until ``register_builtin_benchmarks``
is called, and the empty-registry text used to name only
``register_benchmark_from_file``. An agent asked for a baseline had to find
the built-in registration action by scanning the action enum.

``describe()`` named the built-ins too, from a hand-written list that had gone
stale: ``go2_strafe_left`` and ``go2_turn_left`` shipped after the sentence was
written and were never added to it. Both texts now read
:func:`~strands_robots.simulation.builtin_benchmarks.builtin_benchmark_specs`,
and these checks are keyed off the same source, so a newly bundled benchmark
extends them without an edit.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import benchmark as _benchmark
from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs


class _Engine:
    """The two discovery surfaces under test, off the abstract base.

    ``describe`` reads ``list_robots`` and nothing else, so neither surface
    needs a compiled model - these checks run wherever the registry imports.
    """

    from strands_robots.simulation.base import SimEngine as _Base

    list_benchmarks = _Base.list_benchmarks
    describe = _Base.describe

    def list_robots(self):
        return []


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
    text = _Engine().list_benchmarks()["content"][0]["text"]
    for name in builtin_benchmark_specs():
        assert name in text, f"bundled benchmark {name!r} is not named in the empty-registry text"


def test_describe_names_every_bundled_benchmark_and_its_robot():
    """``describe()`` is the surface an agent reads before calling anything.

    Its ``register_builtin_benchmarks`` entry must name every bundled benchmark
    and the robot each one defaults to, so a caller learns the
    ``benchmark_name`` / ``robot_name`` pair ``evaluate_benchmark`` wants
    without registering first. Both halves are read from the specs, so this
    fails the moment a bundled benchmark is added but the text is not.
    """
    entry = _Engine().describe()["methods"]["register_builtin_benchmarks"]
    specs = builtin_benchmark_specs()

    unnamed = sorted(name for name in specs if name not in entry)
    assert not unnamed, f"bundled benchmarks missing from describe(): {unnamed}"

    robotless = sorted(name for name, spec in specs.items() if spec["default_robot"] not in entry)
    assert not robotless, f"bundled benchmarks named without their default robot: {robotless}"
