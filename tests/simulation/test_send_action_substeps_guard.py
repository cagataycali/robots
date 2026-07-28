"""Regression tests: ``send_action(n_substeps=...)`` is validated like ``step``.

``send_action`` drives the SAME ``mj_step`` loop as ``step`` and holds
``self._lock`` while doing it, but had none of ``step``'s four guards (integer,
non-negative, zero no-op, ``_MAX_STEPS_PER_CALL`` ceiling). Measured:

    n_substeps=10**9  -> no return after 60 s, lock held the whole time
    n_substeps=2.5    -> TypeError: 'float' object cannot be interpreted as an integer
    n_substeps=inf    -> TypeError (same, out of range())
    n_substeps="3"    -> TypeError: '>' not supported between 'str' and 'int'
    n_substeps=0/-1   -> status="success", "Action applied", zero physics steps

The ceiling's own comment is "Hard ceiling to prevent unbounded lock hold", so the
huge-value case blocked every other thread - render, dataset recorder, policy
worker - behind an agent-supplied number. The ``TypeError``s escaped the
``status``/``content`` tool contract entirely.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="send_action_substeps_guard", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    yield s
    s.destroy()


def _send(sim, n_substeps):
    return sim.send_action(action={"joint2": 0.2}, robot_name="a", n_substeps=n_substeps)


@pytest.mark.parametrize("value", ["3", None, [2], {"n": 2}])
def test_non_numeric_substeps_is_a_structured_error(sim, value) -> None:
    result = _send(sim, value)
    assert result["status"] == "error"
    assert "must be an integer" in result["content"][0]["text"]


@pytest.mark.parametrize("value", [2.5, float("nan"), float("inf"), -float("inf")])
def test_non_whole_or_non_finite_substeps_is_rejected(sim, value) -> None:
    result = _send(sim, value)
    assert result["status"] == "error"
    assert "whole finite number" in result["content"][0]["text"]


@pytest.mark.parametrize("value", [0, -1, -100])
def test_substeps_below_one_is_rejected(sim, value) -> None:
    """These reported success while stepping physics zero times."""
    result = _send(sim, value)
    assert result["status"] == "error"
    assert ">= 1" in result["content"][0]["text"]


def test_huge_substeps_is_refused_not_run(sim) -> None:
    """The unbounded-lock-hold case: must return immediately, not loop."""
    result = _send(sim, 10**9)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "exceeds max" in text
    assert str(Simulation._MAX_STEPS_PER_CALL) in text


def test_the_ceiling_matches_step(sim) -> None:
    """Both entry points drive the same loop, so they must share the ceiling."""
    over = Simulation._MAX_STEPS_PER_CALL + 1
    assert _send(sim, over)["status"] == "error"
    assert sim.step(n_steps=over)["status"] == "error"


def test_a_rejected_substep_count_does_not_advance_physics(sim) -> None:
    before = int(sim._world.step_count)
    assert _send(sim, 0)["status"] == "error"
    assert _send(sim, 10**9)["status"] == "error"
    assert int(sim._world.step_count) == before


def test_valid_substeps_still_works(sim) -> None:
    """Guard against the fix degenerating into 'reject everything'."""
    before = int(sim._world.step_count)
    result = _send(sim, 5)
    assert result["status"] == "success"
    assert int(sim._world.step_count) == before + 5


def test_default_substeps_still_works(sim) -> None:
    before = int(sim._world.step_count)
    assert sim.send_action(action={"joint2": 0.2}, robot_name="a")["status"] == "success"
    assert int(sim._world.step_count) == before + 1


def test_numpy_integer_substeps_is_accepted(sim) -> None:
    """A policy loop naturally passes a numpy scalar; it is a whole real number."""
    np = pytest.importorskip("numpy")
    assert _send(sim, np.int64(3))["status"] == "success"


def test_bool_substeps_is_rejected(sim) -> None:
    """``True`` is an int in Python but is never a meaningful substep count."""
    assert _send(sim, True)["status"] == "error"
