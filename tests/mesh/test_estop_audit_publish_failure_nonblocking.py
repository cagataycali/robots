"""Audit-publish failure must never break the e-stop safety path.

``_on_safety_estop`` emits a forensic audit event when it classifies a
remote estop as either a cross-session ``estop_corroborated`` or a replayed
``estop_replay_rejected``. That audit publish is best-effort: the safety
handler documents (and AGENTS.md mandates) that a failing audit sink must
NOT propagate out of the safety path, since doing so would let a flaky or
full-disk audit backend abort estop handling and leave the fleet in a
half-state.

Both audit calls are wrapped in ``except (TypeError, ValueError, OSError)``
so a malformed payload (TypeError/ValueError) or a disk failure (OSError)
is swallowed at DEBUG. These tests pin that contract for both audit
branches: when ``publish_safety_event`` raises one of those errors, the
handler returns cleanly and leaves the lockout latched.
"""

import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import core

_OPERATOR_A_ZID = "0123456789abcdef0123456789abcdef"
_OPERATOR_B_ZID = "fedcba9876543210fedcba9876543210"


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub for safety-handler testing (mirrors test_estop_replay)."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    return m


def _envelope_with_wire_zid(t, peer_id, wire_zid, **extra):
    """Zenoh-sample fake whose source_info zid and body source_zid both equal
    *wire_zid* (production requires body/wire agreement before the cache check).
    """
    body = {"peer_id": peer_id, "t": t, "source_zid": wire_zid, **extra}
    raw = json.dumps(body).encode()

    class _Zid:
        def __init__(self, s):
            self._s = s

        def __str__(self):
            return self._s

    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda r=raw: r),
        source_info=SimpleNamespace(
            source_id=SimpleNamespace(zid=_Zid(wire_zid)),
            source_sn=1,
        ),
    )


def test_corroborated_audit_oserror_is_swallowed_and_lockout_preserved():
    """A disk failure while auditing an estop_corroborated event must not
    propagate out of the safety handler, and the lockout stays latched."""
    mesh = _stub_mesh()
    calls = []

    def raising_audit(**kwargs):
        calls.append(kwargs)
        raise OSError("audit log volume full")

    mesh.publish_safety_event = raising_audit  # type: ignore[method-assign]

    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    mesh._last_estop_mono = time.monotonic()
    # Cache slot bound to operator A's wire zid; operator B corroborates.
    mesh._estop_replay_cache[float(envelope_t)] = (
        "operator-A",
        time.monotonic(),
        _OPERATOR_A_ZID,
    )

    # Must not raise despite the audit sink failing.
    result = mesh._on_safety_estop(
        _envelope_with_wire_zid(
            t=envelope_t,
            peer_id="operator-B",
            wire_zid=_OPERATOR_B_ZID,
            reason="Operator B emergency",
        )
    )

    assert result is None
    # We genuinely reached the corroboration audit branch (and it raised).
    assert len(calls) == 1
    assert calls[0]["event_type"] == "estop_corroborated"
    # Safety state intact: the failing audit did not clear the lockout.
    assert mesh._estop_lockout.is_set()


def test_replay_rejected_audit_valueerror_is_swallowed_and_lockout_preserved():
    """A malformed-payload (ValueError) failure while auditing an
    estop_replay_rejected event must not propagate, and the lockout stays
    latched."""
    mesh = _stub_mesh()
    calls = []

    def raising_audit(**kwargs):
        calls.append(kwargs)
        raise ValueError("bad audit payload shape")

    mesh.publish_safety_event = raising_audit  # type: ignore[method-assign]

    envelope_t = time.time()
    mesh._estop_lockout.set()
    mesh._last_estop_ts = envelope_t
    mesh._last_estop_mono = time.monotonic()
    mesh._estop_replay_cache[float(envelope_t)] = (
        "legit-operator",
        time.monotonic(),
        _OPERATOR_A_ZID,
    )

    # Same wire_zid as the cached slot -> rejection branch (not corroboration).
    result = mesh._on_safety_estop(
        _envelope_with_wire_zid(
            t=envelope_t,
            peer_id="attacker-claims-corroboration",
            wire_zid=_OPERATOR_A_ZID,
            reason="forged corroboration attempt",
        )
    )

    assert result is None
    assert len(calls) == 1
    assert calls[0]["event_type"] == "estop_replay_rejected"
    assert mesh._estop_lockout.is_set()
