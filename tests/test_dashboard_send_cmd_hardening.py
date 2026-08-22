"""send_cmd hardening (DASHBOARD_VS_SDK.md §2.1) — Mesh.send parity.

The bridge's send_cmd was a Mesh.send clone MISSING client-side
validate_command and forged-response rejection: a malformed command went on
the wire, the peer refused it, and the refusal came back as a timeout or
opaque error narrated as a robot fault; and ANY ACL-authorised peer that
observed a turn_id could answer someone else's pending turn.

Run with --no-cov (single-file runs trip the global coverage gate).
"""

from __future__ import annotations

import json
import threading

from strands_robots.dashboard.mesh_bridge import MeshBridge


class FakeSample:
    def __init__(self, data: dict):
        self._raw = json.dumps(data).encode()

    @property
    def payload(self):  # mimics zenoh sample.payload.to_bytes()
        raw = self._raw

        class _P:
            @staticmethod
            def to_bytes() -> bytes:
                return raw

        return _P()


def _bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dashboard-test")
    b._running = True
    return b


class TestClientSideValidation:
    def test_unknown_action_refused_locally_with_the_actual_reason(self):
        b = _bridge()
        res = b.send_cmd("peer-a", {"action": "rm -rf"}, timeout=0.1)
        assert res["ok"] is False
        assert "validation" in res["error"] and "unknown action" in res["error"]

    def test_execute_without_provider_refused_locally(self):
        """The receiver refuses this anyway — but as a timeout-shaped fault.
        Client-side the caller gets the security boundary's own sentence."""
        b = _bridge()
        res = b.send_cmd("peer-a", {"action": "execute", "instruction": "wave"}, timeout=0.1)
        assert res["ok"] is False
        assert "policy_provider" in res["error"]

    def test_refusal_is_recorded_in_activity(self):
        b = _bridge()
        b.send_cmd("peer-a", {"action": "bogus"}, timeout=0.1)
        acts = [a for a in b.activity if a.get("action") == "bogus"]
        assert acts and acts[-1]["ok"] is False

    def test_empty_and_nul_targets_refused(self):
        b = _bridge()
        assert b.send_cmd("", {"action": "status"})["ok"] is False
        assert b.send_cmd("evil\x00peer", {"action": "status"})["ok"] is False


class TestExpectedResponderScope:
    def _pending_turn(self, b: MeshBridge, expected: str) -> str:
        turn = "t" * 32
        with b._rpc_lock:
            b._pending[turn] = threading.Event()
            b._expected_responders[turn] = expected
        return turn

    def test_forged_responder_rejected(self):
        b = _bridge()
        turn = self._pending_turn(b, "peer-a")
        b._on_response(FakeSample({"turn_id": turn, "responder_id": "peer-b", "result": "forged"}))
        assert turn not in b._responses
        assert not b._pending[turn].is_set()

    def test_missing_responder_id_rejected(self):
        """A legacy response with no identity is not a match either."""
        b = _bridge()
        turn = self._pending_turn(b, "peer-a")
        b._on_response(FakeSample({"turn_id": turn, "result": "anonymous"}))
        assert turn not in b._responses

    def test_legitimate_responder_accepted(self):
        b = _bridge()
        turn = self._pending_turn(b, "peer-a")
        b._on_response(FakeSample({"turn_id": turn, "responder_id": "peer-a", "result": "real"}))
        assert b._responses[turn]["result"] == "real"
        assert b._pending[turn].is_set()

    def test_send_cmd_registers_and_clears_expected_responder(self, monkeypatch):
        b = _bridge()
        seen = {}

        def fake_put(topic, envelope):
            seen["topic"] = topic
            with b._rpc_lock:
                seen["expected"] = dict(b._expected_responders)

        monkeypatch.setattr("strands_robots.mesh.session.put", fake_put)
        res = b.send_cmd("peer-a", {"action": "status"}, timeout=0.05)
        assert res["ok"] is False and "timeout" in res["error"]
        assert list(seen["expected"].values()) == ["peer-a"]  # registered before publish
        assert b._expected_responders == {}  # cleared in finally
