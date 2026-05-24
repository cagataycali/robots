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


# === F7-E: per-issuer fairness bound on the estop replay cache ===


class TestEstopPerIssuerFairnessBound:
    """The R20 float-only key closes peer_id-permutation replay but
    opened a denial-of-estop surface where one attacker pre-publishing
    at ``t = now + skew - eps`` could occupy float slots.

    F7-E adds a per-issuer slot cap: each issuer may occupy at most
    ``RESUME_REPLAY_CACHE_MAX // 4`` slots before their newer entries
    are refused. This means at least 4 distinct issuers always have
    working slots regardless of attacker volume.
    """

    def test_one_issuer_cannot_exceed_cap(self, monkeypatch):
        # Tighten the cache size via env (F9-B made this lazy-resolved)
        monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
        # 8 / 4 = 2 slots per issuer

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None  # don't audit in this test

        now = time.time()

        # Attacker fires 5 envelopes at distinct fresh `t` values
        for i in range(5):
            envelope = {"peer_id": "attacker-1", "t": now + 0.001 * i, "type": "estop"}

            class S:
                payload = type("P", (), {"to_bytes": lambda self, e=envelope: json.dumps(e).encode()})()

            m._on_safety_estop(S())

        # F9-A: per-issuer count is derived from cache contents, not a
        # separate dict. Count entries owned by attacker-1; the
        # attacker is capped at per_issuer_cap = MAX // 4 = 2 slots.
        attacker_slots = sum(1 for issuer, _mono in m._estop_replay_cache.values() if issuer == "attacker-1")
        assert attacker_slots <= 2, (
            f"attacker should be capped at 2 slots, got {attacker_slots} (cache: {m._estop_replay_cache})"
        )


# === F9-A: per-issuer count derived from cache contents ===


class TestPerIssuerCountFromCache:
    """F9-A (PR #195 review): the per-issuer fairness bound counts entries
    by issuer from cache contents, not a separate dict that drifts after
    eviction. After eviction, an attacker who flooded their cap legitimately
    has fewer entries and can reclaim slots -- the dynamic-attacker rate
    limit. A sustained attacker is bounded by ``per_issuer_cap`` AT EVERY
    INSTANT, not just between eviction windows.
    """

    def test_cache_carries_issuer_attribution_in_value(self):
        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None

        envelope = {"peer_id": "alice", "t": time.time(), "type": "estop"}

        class S:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        m._on_safety_estop(S())

        assert len(m._estop_replay_cache) == 1
        # Value is now (issuer_id, mono_ts) tuple
        value = next(iter(m._estop_replay_cache.values()))
        assert isinstance(value, tuple), "cache value must be (issuer_id, mono_ts) tuple"
        issuer, mono_ts = value
        assert issuer == "alice"
        assert isinstance(mono_ts, float)


class TestF14OverCapBlocksLockout:
    """F14-B (PR #195 review): when an issuer exceeds per_issuer_cap,
    the lockout MUST NOT engage on their over-cap envelopes. Pre-F14
    the over-cap branch dropped only the cache slot but still let the
    fall-through ``self._estop_lockout.set()`` run, defeating the
    fairness bound's whole point (a sustained attacker at-cap still
    triggered fleet-wide lockouts on every novel ``t`` they emitted).
    """

    def test_at_cap_envelope_does_not_engage_lockout(self, monkeypatch):
        # Tighten cap for fast test
        monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
        # 8 / 4 = 2 slots per issuer

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._running = True
        m.publish = lambda key, data: None
        m.publish_safety_event = lambda **kw: None

        now = time.time()

        # First two envelopes from attacker fill the cap and engage lockout
        for i in range(2):
            envelope = {"peer_id": "attacker", "t": now + 0.001 * i, "type": "estop"}

            class S:
                payload = type("P", (), {"to_bytes": lambda self, e=envelope: json.dumps(e).encode()})()

            m._on_safety_estop(S())

        assert m._estop_lockout.is_set(), "first 2 should engage lockout"

        # Clear lockout and try a 3rd envelope (over cap) -- it must NOT re-engage
        m._estop_lockout.clear()
        envelope = {"peer_id": "attacker", "t": now + 0.005, "type": "estop"}

        class S2:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        m._on_safety_estop(S2())

        assert not m._estop_lockout.is_set(), (
            f"F14-B: over-cap envelope must NOT engage lockout. Cache: {m._estop_replay_cache}"
        )
