"""Regression pins for issue #266: mirror the estop per-issuer fairness
cap in the resume replay cache.

_on_safety_estop bounds any single issuer to
``max(1, _resume_replay_cache_max() // 4)`` cache slots so a compromised
peer cannot fill the global cache and start evicting legitimate
other-issuer entries. Before this fix, _on_safety_resume added to its
replay cache unconditionally, leaving the resume path asymmetrically
unhardened. These tests pin the symmetric behaviour.
"""

from __future__ import annotations

import hmac
import inspect
import json
import time
import uuid
from unittest.mock import MagicMock

from strands_robots.mesh.core import Mesh, _resume_replay_cache_max


def _make_mesh(peer_id="r-test"):
    robot = MagicMock()
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, robot, peer_id)
    return m


def _sample(payload_dict):
    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload_dict).encode()
    return s


def _make_envelope(override_code, *, peer_id="op-1", proof_nonce=None, t=None, lockout_elapsed_s=1.0):
    proof_nonce = proof_nonce or uuid.uuid4().hex
    envelope_t = t if t is not None else time.time()
    mac_input = json.dumps(
        {
            "peer_id": peer_id,
            "t": envelope_t,
            "lockout_elapsed_s": lockout_elapsed_s,
            "proof_nonce": proof_nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    proof = hmac.new(override_code.encode(), mac_input, "sha256").hexdigest()
    return {
        "peer_id": peer_id,
        "t": envelope_t,
        "lockout_elapsed_s": lockout_elapsed_s,
        "proof_nonce": proof_nonce,
        "override_proof": proof,
    }


def test_resume_cache_per_issuer_cap_enforced(monkeypatch):
    """A single issuer filling past the per-issuer cap is refused a cache
    slot and an audit event is emitted."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    # cap = max(1, 8 // 4) = 2
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
    cap = max(1, _resume_replay_cache_max() // 4)
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    for _ in range(cap):
        m._estop_lockout.set()
        m._on_safety_resume(_sample(_make_envelope("secret", peer_id="attacker")))

    # cap slots now held by the attacker
    assert len(m._resume_replay_cache) == cap

    # The next distinct envelope from the SAME issuer must be refused a
    # cache slot (over-cap) -- cache size does not grow.
    m._estop_lockout.set()
    m._on_safety_resume(_sample(_make_envelope("secret", peer_id="attacker")))
    assert len(m._resume_replay_cache) == cap

    cap_audits = [
        c for c in m.publish_safety_event.call_args_list
        if c[1].get("event_type") == "resume_per_issuer_cap_exceeded"
    ]
    assert len(cap_audits) == 1


def test_resume_cache_cap_prevents_eviction_of_other_issuers(monkeypatch):
    """An over-cap issuer cannot add slots, so legitimate other-issuer
    entries already in the cache are never evicted by the attacker's
    churn."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8")
    cap = max(1, _resume_replay_cache_max() // 4)
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # A legitimate operator records one resume.
    m._estop_lockout.set()
    legit_env = _make_envelope("secret", peer_id="operator-legit")
    m._on_safety_resume(_sample(legit_env))
    legit_key = (("body", "operator-legit"), legit_env["proof_nonce"])
    assert legit_key in m._resume_replay_cache

    # Attacker churns many distinct envelopes; capped at ``cap`` slots.
    for _ in range(cap * 5):
        m._estop_lockout.set()
        m._on_safety_resume(_sample(_make_envelope("secret", peer_id="attacker")))

    attacker_slots = sum(
        1 for issuer, _nonce in m._resume_replay_cache if issuer == ("body", "attacker")
    )
    assert attacker_slots <= cap
    # The legitimate entry survived the attacker's churn.
    assert legit_key in m._resume_replay_cache


def test_resume_and_estop_caps_match():
    """Both safety caches derive their per-issuer cap from the identical
    formula, keeping the two handlers symmetric (issue #266)."""
    src = inspect.getsource(Mesh._on_safety_resume)
    estop_src = inspect.getsource(Mesh._on_safety_estop)
    formula = "per_issuer_cap = max(1, _resume_replay_cache_max() // 4)"
    assert formula in src
    assert formula in estop_src
