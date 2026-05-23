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
