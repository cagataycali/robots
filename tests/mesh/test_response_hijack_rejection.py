"""Regression tests for the point-to-point response-scope defence in
``Mesh._on_response``.

Wire authentication (mTLS + ACL) only proves a responder is a fleet
member; it does not prove the responder is the peer the sender
addressed. Without an additional ``responder_id``-vs-expected check, an
ACL-authorised peer that observes a turn_id could publish a response on
someone else's pending turn and have the sender accept its ``result``
instead of the legitimate target's (lateral response hijack).

These tests pin the rejection branch: a point-to-point turn drops a
response whose ``responder_id`` does not match the recorded expected
target, emits the ``response_hijack_rejected`` audit event, and never
records the forged payload -- while broadcast turns still accept any
responder. The audit write is best-effort, so a failing audit log must
not crash the wire-input path nor accept the forged response.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import BROADCAST_RESPONDER


class _FakeRobot:
    """Minimal duck-typed robot; dispatch is not exercised here."""

    def __init__(self) -> None:
        self.tool_name_str = "fakebot"


def _make_sample(payload: dict[str, Any]) -> Any:
    """Build a fake zenoh sample compatible with the mesh callbacks."""
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    return sample


@pytest.fixture
def mesh() -> Mesh:
    """A Mesh instance (not started -- the handlers are driven directly)."""
    return Mesh(_FakeRobot(), peer_id="peer-a", peer_type="robot")


def _register_pending(m: Mesh, turn: str, expected: str) -> threading.Event:
    """Register a pending point-to-point turn the way ``send`` would."""
    event = threading.Event()
    with m._rpc_lock:
        m._pending[turn] = event
        m._responses[turn] = []
        m._expected_responders[turn] = expected
    return event


def test_mismatched_responder_is_dropped(mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    """A response whose responder_id != expected target is not recorded."""
    turn = "turn-point-to-point"
    event = _register_pending(mesh, turn, expected="peer-b")

    sample = _make_sample({"turn_id": turn, "responder_id": "peer-evil", "result": {"ok": 1}})
    with caplog.at_level(logging.WARNING):
        mesh._on_response(sample)

    # The forged payload must NOT be recorded for the turn.
    with mesh._rpc_lock:
        assert mesh._responses[turn] == []
    # The waiter must NOT be released by a hijacked response.
    assert not event.is_set()
    # Operator-facing forensic signal.
    assert any("possible response hijack" in rec.message for rec in caplog.records)


def test_matching_responder_is_recorded(mesh: Mesh) -> None:
    """The legitimate target's response is recorded and releases the waiter."""
    turn = "turn-legit"
    event = _register_pending(mesh, turn, expected="peer-b")

    sample = _make_sample({"turn_id": turn, "responder_id": "peer-b", "result": {"ok": 1}})
    mesh._on_response(sample)

    with mesh._rpc_lock:
        assert mesh._responses[turn] == [{"turn_id": turn, "responder_id": "peer-b", "result": {"ok": 1}}]
    assert event.is_set()


def test_broadcast_turn_accepts_any_responder(mesh: Mesh) -> None:
    """Broadcast turns use the sentinel and accept any responder_id."""
    turn = "turn-broadcast"
    event = _register_pending(mesh, turn, expected=BROADCAST_RESPONDER)

    sample = _make_sample({"turn_id": turn, "responder_id": "anyone", "result": {"i": 1}})
    mesh._on_response(sample)

    with mesh._rpc_lock:
        assert mesh._responses[turn] == [{"turn_id": turn, "responder_id": "anyone", "result": {"i": 1}}]
    assert event.is_set()


def test_hijack_emits_audit_event(mesh: Mesh) -> None:
    """The rejection path logs a structured ``response_hijack_rejected`` event."""
    turn = "turn-audit"
    _register_pending(mesh, turn, expected="peer-b")

    sample = _make_sample({"turn_id": turn, "responder_id": "peer-evil", "result": {"x": 1}})
    with patch.object(mesh_core, "log_safety_event") as mock_audit:
        mesh._on_response(sample)

    mock_audit.assert_called_once()
    args = mock_audit.call_args.args
    assert args[0] == "response_hijack_rejected"
    assert args[1] == "peer-a"
    assert args[2]["responder_id"] == "peer-evil"
    assert args[2]["expected"] == "peer-b"


def test_hijack_audit_failure_is_soft(mesh: Mesh) -> None:
    """A failing audit write must not crash the path nor accept the forgery."""
    turn = "turn-audit-fail"
    event = _register_pending(mesh, turn, expected="peer-b")

    sample = _make_sample({"turn_id": turn, "responder_id": "peer-evil", "result": {"x": 1}})
    with patch.object(mesh_core, "log_safety_event", side_effect=OSError("audit disk full")):
        # Must not raise despite the audit write failing.
        mesh._on_response(sample)

    with mesh._rpc_lock:
        assert mesh._responses[turn] == []
    assert not event.is_set()
