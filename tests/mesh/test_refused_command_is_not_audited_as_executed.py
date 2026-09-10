"""A handler that refuses a command returns a result instead of raising.

``_exec_cmd`` used to audit every non-raising dispatch as ``command_executed``,
so a resume with a bad override code left ``resume_denied`` and
``command_executed action=resume`` in the same audit trail while the lockout
stayed engaged. A refusal is now recorded as ``command_refused`` with the
handler's error, and only a real success is ``command_executed``.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.mesh.core import Mesh


def _mesh(monkeypatch: pytest.MonkeyPatch, result: object) -> tuple[Mesh, list]:
    events: list = []
    monkeypatch.setattr(
        "strands_robots.mesh.core.log_safety_event",
        lambda et, pid, payload: events.append((et, payload)),
    )
    m = Mesh.__new__(Mesh)
    m.peer_id = "robot-1"
    m._cmd_replay_cache = {}
    m._cmd_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._dispatch = lambda cmd: result
    m.publish = lambda *a, **k: None
    return m, events


def _resume(m: Mesh, turn: str = "t1") -> None:
    m._exec_cmd(
        {
            "sender_id": "op",
            "turn_id": turn,
            "command": {"action": "resume", "override_code": "wrong"},
        }
    )


@pytest.mark.parametrize(
    "result",
    [
        {"status": "error", "error": "resume rejected"},
        {"error": "unknown action: resume"},
        {"ok": False},
    ],
    ids=["status-error", "bare-error", "ok-false"],
)
def test_a_refusal_is_audited_as_refused_not_executed(monkeypatch, result):
    m, events = _mesh(monkeypatch, result)

    _resume(m)

    names = [et for et, _ in events]
    assert "command_executed" not in names, events
    assert names == ["command_refused"], events
    payload = events[0][1]
    assert payload["action"] == "resume"
    assert payload["turn_id"] == "t1"
    assert payload["error"] == result.get("error", "ok=False")


def test_a_success_is_still_audited_as_executed(monkeypatch):
    m, events = _mesh(monkeypatch, {"status": "resumed", "lockout_cleared": True})

    _resume(m)

    assert [et for et, _ in events] == ["command_executed"], events
    assert "error" not in events[0][1]


def test_a_refused_readonly_command_is_not_audited(monkeypatch):
    m, events = _mesh(monkeypatch, {"error": "no status available"})

    m._exec_cmd({"sender_id": "op", "turn_id": "t3", "command": {"action": "status"}})

    assert events == []
