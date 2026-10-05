"""Guard: MuJoCo backend's get_observation() emits a WARNING naming the
remedy before returning the empty-dict degraded mode.

Three sibling methods on the same engine (``send_action``, ``step``,
``get_robot_state``) route through ``_NO_WORLD_MSG`` and surface
``status="error"`` on the same condition - the empty-dict is kept here to
preserve the dict signature across the three-backend contract (Isaac /
MuJoCo / Newton), but the log is what keeps "empty" from being silent on a
rollout that reads :meth:`get_observation` as its heartbeat. Pinned so a
future refactor cannot drop the WARNING back to a bare ``return {}``.

Companion to Isaac's existing invariant at
``simulation/isaac/simulation.py:5639-5667`` (five named conditions, each
with its own ``logger.warning``) - the MuJoCo backend was the odd one out
before this fix.
"""
from __future__ import annotations

import logging

import pytest

from strands_robots import Robot


_MOD = "strands_robots.simulation.mujoco.simulation"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == _MOD and rec.levelno == logging.WARNING
    ]


def test_get_observation_logs_warning_on_torn_down_world(caplog):
    """After destroy(), get_observation() returns {} AND logs the remedy."""
    r = Robot("so101")
    assert r.get_observation(), "pre-destroy obs must be non-empty"
    destroy_res = r.destroy()
    assert destroy_res["status"] == "success"

    with caplog.at_level(logging.WARNING, logger=_MOD):
        obs = r.get_observation()

    assert obs == {}, f"torn-down world must return empty dict (got {obs!r})"
    warns = _warnings(caplog)
    assert warns, "get_observation() on torn-down world emitted NO warning"
    assert any("no world" in w.lower() for w in warns), (
        f"warning does not name the remedy: {warns!r}"
    )


def test_get_observation_logs_warning_on_unknown_robot(caplog):
    """A typo'd robot_name returns {} AND names the known roster in the log."""
    r = Robot("so101")

    with caplog.at_level(logging.WARNING, logger=_MOD):
        obs = r.get_observation(robot_name="nonexistent_bot")

    assert obs == {}
    warns = _warnings(caplog)
    assert warns, "get_observation() on unknown robot emitted NO warning"
    joined = " ".join(warns).lower()
    assert "unknown robot" in joined
    assert "so101" in joined, (
        f"warning does not name the known roster so the caller can self-correct: {warns!r}"
    )


def test_get_observation_still_returns_dict_shape(caplog):
    """Shape contract preserved: degraded mode is {}, never None, never raises.

    Pinned so a future "raise instead of return {}" refactor does not break
    the two-backend-batched rollout that reads obs across engines.
    """
    r = Robot("so101")
    r.destroy()
    with caplog.at_level(logging.WARNING, logger=_MOD):
        obs = r.get_observation()
    assert isinstance(obs, dict)
    assert obs == {}
