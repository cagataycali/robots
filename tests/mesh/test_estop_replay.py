"""
Pin test for estop corroboration attribution (R22-B).

When two distinct operators broadcast safety/estop at colliding t values,
the audit log should record this as estop_corroborated (positive forensic)
not estop_replay_rejected (false negative).
"""

import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import audit as audit_mod
from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub for safety handler testing."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    return m


def _envelope(t: float, peer_id: str = "issuer", **extra):
    body = {"peer_id": peer_id, "t": t, **extra}
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def test_distinct_issuers_same_t_audited_as_corroborated():
    """
    Two distinct operators with colliding t should audit as corroborated.

    Pre-fix: second estop audited as estop_replay_rejected (false negative).
    Post-fix: second estop within 0.2s of lockout audited as estop_corroborated.
    """
    mesh = _stub_mesh()
    audit_calls = []

    def capture_audit(**kwargs):
        audit_calls.append(kwargs)

    mesh.publish_safety_event = capture_audit  # type: ignore[method-assign]

    # First estop from operator A
    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    mesh._estop_replay_cache[float(envelope_t)] = time.monotonic()

    # Second estop from operator B with same t (colliding timestamp)
    mesh._on_safety_estop(_envelope(t=envelope_t, peer_id="operator-B", reason="Operator B emergency"))

    # Pre-fix: audit_calls contains estop_replay_rejected
    # Post-fix: audit_calls contains estop_corroborated (lockout active, within 0.2s)
    assert len(audit_calls) == 1, f"Expected 1 audit call, got {len(audit_calls)}"
    assert audit_calls[0]["event_type"] == "estop_corroborated", (
        f"Expected estop_corroborated, got {audit_calls[0]['event_type']}"
    )
    assert audit_calls[0]["severity"] == "info"
    assert audit_calls[0]["payload"]["issuer"] == "operator-B"


class TestEstopRedundantAudit:
    """When a second-operator estop arrives while lockout is already
    engaged, an audit event must be emitted (forensic preservation).
    """

    def test_redundant_estop_emits_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        # Reset audit state for isolated test
        audit_mod._AUDIT_STATE.psk_fingerprint = None
        audit_mod._AUDIT_STATE.seq_loaded = False
        audit_mod._SEQ_COUNTERS.clear()

        class StubRobot:
            pass

        m = core.Mesh(robot=StubRobot(), peer_id="robot-r")
        # publish_safety_event is gated on self._running; flip it on
        # without calling start() (which does network I/O). Stub publish()
        # since we only care about the audit-log side-effect.
        m._running = True
        m.publish = lambda key, data: None
        # First estop engages
        e1 = {"peer_id": "op-1", "t": time.time(), "type": "estop"}

        class S:
            def __init__(self, e):
                self.payload = type("P", (), {"to_bytes": lambda self: json.dumps(e).encode()})()

        m._on_safety_estop(S(e1))
        assert m._estop_lockout.is_set()

        # Second-operator estop, fresh `t`, lockout already engaged
        e2 = {"peer_id": "op-2", "t": time.time() + 0.5, "type": "estop"}
        m._on_safety_estop(S(e2))

        # Walk the audit log
        records = audit_mod.read_audit_log()
        events = [r["event"] for r in records]
        assert "remote_estop_engaged" in events, f"first engagement missing: {events}"
        assert "remote_estop_redundant" in events, f"second-operator audit missing: {events}"


# ---------------------------------------------------------------------
# F3-C-1: _PSK_STATE_LOCK exists and protects fingerprint snapshot
# ---------------------------------------------------------------------
