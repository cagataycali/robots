"""A recorded frame's timestamp is the simulation time actually advanced (#4392).

On MuJoCo's defaults (``dt=0.002``) a 30 Hz control period is 16.67 physics
steps. Rounding it once to 17 made every action advance 0.034 s while LeRobot
stamped the frame ``k / 30`` s: 60 frames spanned 1.967 s of timestamps over
2.040 s of simulation, 2 percent early on the default recording path. The
runner now schedules substeps cumulatively (17, 17, 16, ...) so the k-th
action starts within one physics step of ``k / control_frequency``, and a
rollout of ``n`` actions covers ``n / control_frequency`` of sim time to within
one step. An exact division is unchanged: a constant count, no warning.
"""

from __future__ import annotations

import logging
import types
from typing import Any, cast

import pytest

from strands_robots.simulation.policy_runner import PolicyRunner

_LOGGER = "strands_robots.simulation.policy_runner"


def _runner(dt: float) -> PolicyRunner:
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.sim = cast(Any, types.SimpleNamespace(physics_timestep=lambda: dt))
    return runner


def test_sixty_frames_at_30_hz_cover_two_seconds_of_2ms_steps() -> None:
    """The issue's numbers: 60 actions cover 1000 steps (2.000 s), not 1020 (2.040 s)."""
    schedule = _runner(0.002)._substep_schedule(30.0)
    counts = [schedule.next() for _ in range(60)]
    assert set(counts) == {16, 17}
    assert sum(counts) == 1000
    # Every frame boundary lands within one physics step of k / 30 s.
    elapsed = 0
    for k, count in enumerate(counts):
        assert abs(elapsed * 0.002 - k / 30.0) <= 0.002, (k, elapsed)
        elapsed += count


def test_the_isaac_default_alternates_two_and_three_steps() -> None:
    """1/120 s against 50 Hz is 2.4 steps: five actions cover 12 steps (0.100 s), not 10."""
    schedule = _runner(1 / 120)._substep_schedule(50.0)
    counts = [schedule.next() for _ in range(5)]
    assert set(counts) == {2, 3}
    assert sum(counts) == 12
    assert schedule.nominal == 2 and schedule.exact is False


@pytest.mark.parametrize(("dt", "hz", "expected"), [(0.002, 50.0, 10), (1 / 100, 50.0, 2), (1 / 120, 30.0, 4)])
def test_an_exact_division_is_a_constant_count(caplog, dt: float, hz: float, expected: int) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        schedule = _runner(dt)._substep_schedule(hz)
    assert schedule.exact is True
    assert [schedule.next() for _ in range(25)] == [expected] * 25
    assert not [r for r in caplog.records if "physics steps" in r.getMessage()]


def test_an_override_is_honoured_verbatim() -> None:
    schedule = _runner(0.002)._substep_schedule(30.0, 5)
    assert [schedule.next() for _ in range(4)] == [5, 5, 5, 5]


def test_an_unknown_timestep_is_one_step_per_action() -> None:
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.sim = cast(Any, types.SimpleNamespace(physics_timestep=lambda: None))
    schedule = runner._substep_schedule(30.0)
    assert [schedule.next() for _ in range(3)] == [1, 1, 1]


def test_the_inexact_period_is_still_said_once_with_the_effective_numbers(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _runner(0.002)._substep_schedule(30.0)
    [record] = [r for r in caplog.records if "physics steps" in r.getMessage()]
    message = record.getMessage()
    assert "16.67 physics steps" in message
    assert "16 and 17" in message
    assert "within one physics step" in message
