"""A control period that is not a whole number of physics steps is reported.

``PolicyRunner`` steps ``round(period / physics_dt)`` physics steps per action.
On Isaac's default ``physics_dt=1/120`` against the default 50 Hz that is 2.4 ->
2, so every action advanced 16.7 ms instead of 20 ms: ``run_policy(n_steps=30)``
ended at ``sim_t=0.517 s`` where MuJoCo (2 ms dt, exactly 10 substeps) ended at
0.600 s (measured on a live GPU run), and a recording labelled 50 fps held
frames 1/60 s apart in sim time. Nothing said so.
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


def test_the_isaac_default_rounds_and_says_so(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _runner(1 / 120)._control_substeps(50.0) == 2

    [record] = [r for r in caplog.records if "rounding to 2" in r.getMessage()]
    message = record.getMessage()
    assert "60 Hz" in message and "not 50 Hz" in message


@pytest.mark.parametrize(("dt", "hz", "expected"), [(0.002, 50.0, 10), (1 / 100, 50.0, 2), (1 / 120, 30.0, 4)])
def test_an_exact_division_is_silent(caplog, dt: float, hz: float, expected: int) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _runner(dt)._control_substeps(hz) == expected
    assert not [r for r in caplog.records if "rounding" in r.getMessage()]
