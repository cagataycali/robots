"""Regression tests for resume-replay defenses in Mesh._on_safety_resume.

This test suite validates the fix for PR #195 reviewer concern:
the cryptographic shape `HMAC(override_code, proof_nonce)` with the issuer
choosing the nonce originally gave no replay defense between fellow
operator-class peers.

The fix in core.py adds:
1. Freshness window (RESUME_FRESHNESS_WINDOW_S, default 60s)
2. Forward-skew bound (RESUME_FORWARD_SKEW_S, default 5s)
3. Per-receiver replay cache ((issuer_peer_id, proof_nonce) tuple)
4. Bounded cache (RESUME_REPLAY_CACHE_MAX, default 4096) with stale-entry
   sweep + oldest-20%-drop fallback
"""

import hmac
import json
import logging
import time
import uuid
from unittest.mock import MagicMock

from strands_robots.mesh import core as core_module
from strands_robots.mesh.core import Mesh


def _make_mesh(peer_id="r-test"):
    """Construct a minimally-instantiated Mesh without calling init_mesh."""
    robot = MagicMock()
    m = Mesh.__new__(Mesh)  # bypass __init__ side-effects
    Mesh.__init__(m, robot, peer_id)
    return m


def _sample(payload_dict):
    """Make a fake zenoh sample with a JSON payload."""
    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload_dict).encode()
    return s


def _make_envelope(override_code, *, t=None, peer_id="op-1", proof_nonce=None):
    """Mint a valid resume envelope keyed off a specific override code."""
    proof_nonce = proof_nonce or uuid.uuid4().hex
    proof = hmac.new(override_code.encode(), proof_nonce.encode(), "sha256").hexdigest()
    return {
        "peer_id": peer_id,
        "t": t if t is not None else time.time(),
        "lockout_elapsed_s": 1.0,
        "proof_nonce": proof_nonce,
        "override_proof": proof,
    }


def test_first_legitimate_resume_clears_lockout(monkeypatch):
    """Sanity / happy path: a valid fresh envelope clears the lockout."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()  # stub out audit publishing

    # Set lockout
    m._estop_lockout.set()
    assert m._estop_lockout.is_set()

    # Send valid resume
    env = _make_envelope("secret")
    m._on_safety_resume(_sample(env))

    # Lockout should be cleared
    assert m._estop_lockout.is_set() is False


def test_replay_of_same_envelope_is_rejected(monkeypatch):
    """Replay of the same envelope is rejected via cache."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint ONE envelope
    env = _make_envelope("secret", peer_id="op-1")

    # First resume: accepted
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is False

    # Re-arm lockout
    m._estop_lockout.set()

    # Second resume with SAME envelope: rejected (replay)
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is True  # lockout stays set

    # Verify cache contains the (issuer, proof_nonce) tuple
    cache_key = (env["peer_id"], env["proof_nonce"])
    assert cache_key in m._resume_replay_cache

    # Verify audit event was emitted
    calls = [c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "resume_replay_rejected"]
    assert len(calls) == 1


def test_stale_envelope_rejected(monkeypatch):
    """Envelope older than RESUME_FRESHNESS_WINDOW_S is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 1 hour ago
    env = _make_envelope("secret", t=time.time() - 3600)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set (envelope rejected for staleness)
    assert m._estop_lockout.is_set() is True


def test_future_envelope_rejected_beyond_skew(monkeypatch):
    """Envelope with t beyond RESUME_FORWARD_SKEW_S is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 60s in future (beyond default 5s skew)
    env = _make_envelope("secret", t=time.time() + 60)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_envelope_within_forward_skew_accepted(monkeypatch):
    """Envelope with t within RESUME_FORWARD_SKEW_S is accepted."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with t = 1s in future (within default 5s skew)
    env = _make_envelope("secret", t=time.time() + 1.0)

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should be cleared
    assert m._estop_lockout.is_set() is False


def test_replay_cache_bounded(monkeypatch):
    """Replay cache is bounded at RESUME_REPLAY_CACHE_MAX."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")

    # Lower the cap for testing
    original_max = core_module.RESUME_REPLAY_CACHE_MAX
    monkeypatch.setattr(core_module, "RESUME_REPLAY_CACHE_MAX", 8)

    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Drive 20 distinct nonces through the resume handler
    for i in range(20):
        env = _make_envelope("secret", peer_id=f"op-{i}", proof_nonce=uuid.uuid4().hex)
        m._estop_lockout.set()
        m._on_safety_resume(_sample(env))

    # Cache should be bounded
    assert len(m._resume_replay_cache) <= 8

    # Restore original
    monkeypatch.setattr(core_module, "RESUME_REPLAY_CACHE_MAX", original_max)


def test_envelope_missing_t_field_rejected(monkeypatch):
    """Envelope missing the t field is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope then delete t field
    env = _make_envelope("secret")
    del env["t"]

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_envelope_invalid_t_type_rejected(monkeypatch):
    """Envelope with invalid t type is rejected."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint envelope with invalid t type
    env = _make_envelope("secret")
    env["t"] = "not-a-number"

    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))

    # Lockout should still be set
    assert m._estop_lockout.is_set() is True


def test_replay_emits_audit_event(monkeypatch):
    """Replay rejection emits resume_replay_rejected audit event."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret")
    m = _make_mesh()
    m.publish_safety_event = MagicMock()

    # Mint ONE envelope
    env = _make_envelope("secret", peer_id="op-attacker")

    # First resume: accepted
    m._estop_lockout.set()
    m._on_safety_resume(_sample(env))
    assert m._estop_lockout.is_set() is False

    # Check that resume_clear event was emitted
    clear_calls = [
        c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "remote_resume_applied"
    ]
    assert len(clear_calls) == 1

    # Re-arm lockout
    m._estop_lockout.set()

    # Second resume with SAME envelope: replay rejected
    m._on_safety_resume(_sample(env))

    # Check that resume_replay_rejected event was emitted
    replay_calls = [
        c for c in m.publish_safety_event.call_args_list if c[1].get("event_type") == "resume_replay_rejected"
    ]
    assert len(replay_calls) == 1

    # Verify payload contains issuer
    replay_payload = replay_calls[0][1]["payload"]
    assert replay_payload["issuer"] == "op-attacker"
    assert "proof_nonce_prefix" in replay_payload


class TestResumeStrictPeerId:
    """The estop handler (R20) rejects envelopes with empty/missing
    peer_id outright. Resume must mirror that posture.
    """

    def test_resume_with_empty_peer_id_rejected(self, caplog):
        class StubRobot:
            pass

        m = Mesh(robot=StubRobot(), peer_id="robot-test")
        m._estop_lockout.set()  # lockout is engaged so resume would normally clear it

        # Forge a resume envelope that's otherwise well-formed but has empty peer_id
        envelope = {
            "peer_id": "",
            "t": time.time(),
            "proof_nonce": "n" * 32,
            "override_proof": "x" * 64,
        }

        class FakeSample:
            payload = type("P", (), {"to_bytes": lambda self: json.dumps(envelope).encode()})()

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            m._on_safety_resume(FakeSample())

        # Lockout MUST still be engaged (resume rejected)
        assert m._estop_lockout.is_set(), "resume with empty peer_id should NOT clear lockout"
        # Cache should NOT have a polluting entry
        assert len(m._resume_replay_cache) == 0, "no cache entry should be created for invalid peer_id"


# ---------------------------------------------------------------------
# F3-B-4: remote_estop_redundant audit on second-operator estop
# ---------------------------------------------------------------------
