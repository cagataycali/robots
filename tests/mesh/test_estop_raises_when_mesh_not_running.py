"""An e-stop on a mesh that never started must raise, not return ``[]``.

``[]`` is also what a healthy broadcast returns when no peer answers, so a
caller (or an operator reading the tool output) could not tell "every robot
was told to stop and none replied" from "nothing was sent at all".
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.mesh.core import Mesh


class _FakeRobot:
    robot_type = "fake"


def _unstarted_mesh() -> Mesh:
    m = Mesh(_FakeRobot(), peer_id="never-started")
    assert m.alive is False
    return m


def test_emergency_stop_raises_and_does_not_engage_lockout(caplog: pytest.LogCaptureFixture) -> None:
    m = _unstarted_mesh()
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(RuntimeError, match="mesh not running"):
            m.emergency_stop()
    assert not m._estop_lockout.is_set()
    assert "EMERGENCY STOP engaged" not in caplog.text


def test_broadcast_raises_and_names_the_action() -> None:
    m = _unstarted_mesh()
    with pytest.raises(RuntimeError, match="cannot broadcast stop"):
        m.broadcast({"action": "stop"}, timeout=0.1)
