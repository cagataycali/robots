"""A degraded ``_read_state`` probe is reported, exactly once per category.

``Mesh._read_state`` probes a robot driver defensively so that a flaky read
cannot kill the state thread. Historically each probe ended in a bare
``except Exception: pass``, so a broken joint read simply stopped publishing
joints and left no trace anywhere -- the failure mode recorded as bug #3 in
``BUGS.md``. These tests pin the contract that replaced it:

* the snapshot still publishes every section that *did* work (unchanged
  return semantics), and
* the first failure of each category is logged at WARNING with the
  exception's ``repr`` and the peer id, while later ticks stay quiet
  (the loop retries at ``STATE_HZ``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import Mesh

CORE_LOGGER = "strands_robots.mesh.core"


class _BrokenJointReadInner:
    """A connected hardware driver whose ``get_observation`` always raises."""

    is_connected = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(cameras={})
        self.calls = 0

    def get_observation(self) -> dict[str, Any]:
        """Fail the way a wedged serial bus does: loudly, every time."""
        self.calls += 1
        raise RuntimeError("joint bus read failed")


class _BrokenJointRobot:
    """A robot whose joints are unreadable but whose task state is fine."""

    def __init__(self) -> None:
        self.robot = _BrokenJointReadInner()
        self._task_state = SimpleNamespace(
            status="running",
            instruction="pick up the cube",
            step_count=3,
            duration=1.5,
        )


class _BrokenWorldRobot:
    """A sim-shaped robot whose ``_world`` back-reference raises on access."""

    def __init__(self) -> None:
        self._task_state = SimpleNamespace(
            status="running",
            instruction="hold",
            step_count=1,
            duration=0.5,
        )

    @property
    def _world(self) -> Any:
        raise RuntimeError("world unavailable")


def _core_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the WARNING records emitted by the mesh core logger."""
    return [r for r in caplog.records if r.name == CORE_LOGGER and r.levelno == logging.WARNING]


class TestReadStateDegradationIsLogged:
    """A failing probe warns once, and the rest of the snapshot survives."""

    def test_hardware_joint_read_failure_warns_once_and_still_publishes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two ticks with a broken joint read -> one warning, task still sent."""
        robot = _BrokenJointRobot()
        m = Mesh(robot, peer_id="broken-joints")

        with caplog.at_level(logging.WARNING, logger=CORE_LOGGER):
            first = m._read_state()
            second = m._read_state()

        # Return semantics are unchanged: a snapshot is still published, it
        # simply has no joints section.
        for snapshot in (first, second):
            assert snapshot is not None
            assert "joints" not in snapshot
            assert snapshot["peer_id"] == "broken-joints"
            assert snapshot["task"]["instruction"] == "pick up the cube"

        # The probe really was attempted on both ticks (no silent latching).
        assert robot.robot.calls == 2

        warnings = _core_warnings(caplog)
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        assert "hw_joints" in message
        assert "broken-joints" in message
        assert "RuntimeError('joint bus read failed')" in message

    def test_repeat_failures_are_kept_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """After the first warning, later failures are debug-level, not silent."""
        m = Mesh(_BrokenJointRobot(), peer_id="quiet-after-first")

        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            m._read_state()
            caplog.clear()
            m._read_state()

        assert _core_warnings(caplog) == []
        debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("hw_joints" in msg and "still failing" in msg for msg in debug_messages)

    def test_categories_warn_independently(self, caplog: pytest.LogCaptureFixture) -> None:
        """A sim-world failure is its own category, so it is not masked."""
        m = Mesh(_BrokenWorldRobot(), peer_id="broken-world")

        with caplog.at_level(logging.WARNING, logger=CORE_LOGGER):
            snapshot = m._read_state()
            m._read_state()

        assert snapshot is not None
        assert "task" in snapshot
        warnings = _core_warnings(caplog)
        assert len(warnings) == 1
        assert "sim_world" in warnings[0].getMessage()
        assert m._read_state_warned == {"sim_world"}
