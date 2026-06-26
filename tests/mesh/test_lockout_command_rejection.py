"""Emergency-stop lockout command-rejection wire + audit contract.

When the e-stop lockout is engaged, ``Mesh._dispatch`` raises
``LockoutError`` for any actuating action (everything except ``status`` and
``resume``). ``Mesh._exec_cmd`` must translate that into the same observable
shape as every other rejection on the security boundary:

  * a ``type="error"`` envelope on the requester's response topic whose error
    string is the deliberately generic ``"command rejected"`` -- a remote
    caller must not be able to use the wire response to map the lockout
    window, so the message carries no lockout-specific detail; and
  * a ``command_rejected_lockout`` audit record carrying the sender and the
    rejected action, so post-incident forensics can see actuation attempts
    made while the fleet was halted.

``status`` and ``resume`` are exempt so an operator can still poll state and
lift the lockout while it is engaged; a ``status`` probe during lockout must
flow through to a normal response and emit no rejection audit.

These pin the lockout branch of ``_exec_cmd`` which was otherwise only
reachable through a live Zenoh e-stop broadcast.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import core
from strands_robots.mesh.core import Mesh


class _FakeRobot:
    """Minimal robot adapter with a readonly status surface."""

    def status(self) -> dict[str, Any]:
        return {"status": "idle"}

    def stop_task(self) -> dict[str, Any]:
        return {"ok": True}


def _capture(m: Mesh, monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Capture outbound publishes and audit events on *m* without I/O."""
    puts: list[tuple[str, dict]] = []
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(m, "publish", lambda key, payload, **kw: puts.append((key, payload)))
    monkeypatch.setattr(
        core, "log_safety_event", lambda event_type, peer_id, detail: events.append((event_type, detail))
    )
    return puts, events


def test_lockout_rejects_actuating_command_with_generic_wire_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An actuating command during lockout gets a generic error on its response topic."""
    m = Mesh(_FakeRobot(), peer_id="robot-a")
    puts, _ = _capture(m, monkeypatch)
    m._estop_lockout.set()

    m._exec_cmd({"sender_id": "op1", "turn_id": "t1", "command": {"action": "stop"}})

    responses = [payload for key, payload in puts if key == "strands/op1/response/robot-a/t1"]
    assert responses, puts
    err = responses[0]
    assert err["type"] == "error"
    assert err["responder_id"] == "robot-a"
    assert err["turn_id"] == "t1"
    # Deliberately generic: a remote caller must not be able to map the
    # lockout window from the wire response.
    assert err["error"] == "command rejected"


def test_lockout_rejection_is_audited_with_sender_and_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lockout rejection records a forensic command_rejected_lockout event."""
    m = Mesh(_FakeRobot(), peer_id="robot-a")
    _, events = _capture(m, monkeypatch)
    m._estop_lockout.set()

    m._exec_cmd({"sender_id": "intruder", "turn_id": "t9", "command": {"action": "stop"}})

    lockout_events = [detail for event_type, detail in events if event_type == "command_rejected_lockout"]
    assert lockout_events == [{"sender": "intruder", "action": "stop"}], events


def test_lockout_permits_status_poll_without_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """status is exempt from lockout: it returns a response and emits no rejection audit."""
    m = Mesh(_FakeRobot(), peer_id="robot-a")
    puts, events = _capture(m, monkeypatch)
    m._estop_lockout.set()

    m._exec_cmd({"sender_id": "op1", "turn_id": "t1", "command": {"action": "status"}})

    responses = [payload for _, payload in puts]
    assert responses, puts
    assert responses[0]["type"] == "response"
    assert not any(event_type == "command_rejected_lockout" for event_type, _ in events), events


def test_lockout_wire_error_survives_audit_sink_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing audit sink must not break the lockout rejection response.

    Audit emission is best-effort: ``log_safety_event`` can raise on payload
    shape (TypeError/ValueError) or disk failure (OSError). The lockout
    rejection still has to land its ``type="error"`` response on the wire --
    a broken audit log cannot be allowed to swallow a rejection and let the
    actuating command appear to succeed.
    """

    def _boom(event_type: str, peer_id: str, detail: dict[str, Any]) -> None:
        raise OSError("audit disk full")

    m = Mesh(_FakeRobot(), peer_id="robot-a")
    puts: list[tuple[str, dict]] = []
    monkeypatch.setattr(m, "publish", lambda key, payload, **kw: puts.append((key, payload)))
    monkeypatch.setattr(core, "log_safety_event", _boom)
    m._estop_lockout.set()

    # Must not raise despite the audit sink failing.
    m._exec_cmd({"sender_id": "op1", "turn_id": "t1", "command": {"action": "stop"}})

    errors = [payload for key, payload in puts if payload.get("type") == "error"]
    assert errors, puts
    assert errors[0]["error"] == "command rejected"
