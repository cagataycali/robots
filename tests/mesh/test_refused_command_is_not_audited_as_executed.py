"""A handler that refuses a command returns a result instead of raising.

``_exec_cmd`` used to audit every non-raising dispatch as ``command_executed``,
so a resume with a bad override code left ``resume_denied`` and
``command_executed action=resume`` in the same audit trail while the lockout
stayed engaged. A refusal is now recorded as ``command_refused`` carrying the
handler's error, and only a real success is ``command_executed``.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.mesh.core import Mesh

#: The captured ``(event_type, payload)`` pairs a turn wrote to the audit log.
_Events = list[tuple[str, dict[str, Any]]]


def _mesh(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> tuple[Mesh, _Events]:
    """A Mesh whose dispatch answers ``result``, with its audit log captured."""
    events: _Events = []
    monkeypatch.setattr(
        "strands_robots.mesh.core.log_safety_event",
        lambda et, pid, payload: events.append((et, payload)),
    )
    m = Mesh.__new__(Mesh)
    m.peer_id = "robot-1"
    m._cmd_replay_cache = {}
    m._cmd_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    monkeypatch.setattr(m, "_dispatch", lambda cmd: result)
    monkeypatch.setattr(m, "publish", lambda *a, **k: None)
    return m, events


def _resume(m: Mesh, turn: str = "t1") -> None:
    """Send the drill's own command: a resume carrying a bad override code."""
    m._exec_cmd(
        {
            "sender_id": "op",
            "turn_id": turn,
            "command": {"action": "resume", "override_code": "wrong"},
        }
    )


@pytest.mark.parametrize(
    ("result", "expected_error"),
    [
        pytest.param({"status": "error", "error": "resume rejected"}, "resume rejected", id="status-error"),
        pytest.param({"error": "unknown action: resume"}, "unknown action: resume", id="bare-error"),
        pytest.param({"ok": False}, "ok=False", id="ok-false"),
        pytest.param(
            {"status": "error", "content": [{"text": "resume rejected: bad override code"}]},
            "status=error",
            id="tool-envelope-carries-no-error-key",
        ),
    ],
)
def test_a_refusal_is_audited_as_refused_not_executed(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], expected_error: str
) -> None:
    m, events = _mesh(monkeypatch, result)

    _resume(m)

    names = [et for et, _ in events]
    assert "command_executed" not in names, events
    assert names == ["command_refused"], events
    payload = events[0][1]
    assert payload["action"] == "resume", events
    assert payload["turn_id"] == "t1", events
    assert payload["error"] == expected_error, events


def test_a_success_is_still_audited_as_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    m, events = _mesh(monkeypatch, {"status": "resumed", "lockout_cleared": True})

    _resume(m)

    assert [et for et, _ in events] == ["command_executed"], events
    assert "error" not in events[0][1], events


def test_a_refused_readonly_command_is_not_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    m, events = _mesh(monkeypatch, {"error": "no status available"})

    m._exec_cmd({"sender_id": "op", "turn_id": "t3", "command": {"action": "status"}})

    assert events == [], events
