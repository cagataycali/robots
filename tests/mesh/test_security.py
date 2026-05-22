"""Unit tests for :mod:`strands_robots.mesh.security`.

Covers the HMAC envelope sign/verify round-trip, replay-window and nonce
behaviour, command validation rules (action allowlist, instruction bounds,
``policy_host`` allowlist, parameter coercion), the per-sender token-bucket
rate limiter, and the various env-var clamps that defend against operator
misconfiguration.
"""

from __future__ import annotations

import time

import pytest

from strands_robots.mesh import security as sec


@pytest.fixture(autouse=True)
def _isolate_security(monkeypatch):
    """Each test runs with a clean nonce cache and known PSK config."""
    monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REPLAY_WINDOW", raising=False)
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    sec._PSK_WARNED = False  # reset one-shot warning so tests are deterministic
    sec.clear_replay_cache()
    yield
    sec.clear_replay_cache()


# ─── HMAC envelope sign / verify ────────────────────────────────────────


class TestEnvelope:
    def test_sign_envelope_unsigned_when_no_psk(self):
        env = sec.sign_envelope({"action": "status"})
        assert env["v"] == 1
        assert "ts" in env and "nonce" in env
        assert env["payload"] == {"action": "status"}
        assert "sig" not in env

    def test_sign_envelope_with_psk(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "topsecret")
        env = sec.sign_envelope({"action": "status"})
        assert "sig" in env
        assert len(env["sig"]) == 64  # sha256 hex

    def test_verify_round_trip(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "topsecret")
        env = sec.sign_envelope({"action": "status"})
        assert sec.verify_envelope(env) == {"action": "status"}

    def test_verify_rejects_tampered_payload(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "topsecret")
        env = sec.sign_envelope({"action": "status"})
        env["payload"]["action"] = "execute"  # tamper
        with pytest.raises(sec.AuthenticationError, match="HMAC"):
            sec.verify_envelope(env)

    def test_verify_rejects_tampered_signature(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "topsecret")
        env = sec.sign_envelope({"action": "status"})
        env["sig"] = "0" * 64
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope(env)

    def test_verify_rejects_wrong_psk(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "alpha")
        env = sec.sign_envelope({"action": "status"})
        monkeypatch.setenv("STRANDS_MESH_PSK", "beta")
        sec.clear_replay_cache()
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope(env)

    def test_verify_legacy_passthrough_in_permissive_mode(self):
        # Bare dict (no envelope) — accepted in permissive mode for back-compat.
        bare = {"action": "status", "sender_id": "a"}
        assert sec.verify_envelope(bare) == bare

    def test_strict_mode_rejects_bare_dict(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_REQUIRE_AUTH", "true")
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope({"action": "status"})

    def test_strict_mode_requires_psk_for_signed_envelope(self, monkeypatch):
        # No PSK + strict → reject envelopes that pretend to be signed
        monkeypatch.setenv("STRANDS_MESH_REQUIRE_AUTH", "true")
        env = sec.sign_envelope({"action": "status"})  # produced w/o PSK
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope(env)

    def test_unknown_envelope_version(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "topsecret")
        env = sec.sign_envelope({"action": "status"})
        env["v"] = 99
        with pytest.raises(sec.AuthenticationError, match="version"):
            sec.verify_envelope(env)

    def test_envelope_input_must_be_dict(self):
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope("not a dict")  # type: ignore[arg-type]

    def test_sign_input_must_be_dict(self):
        with pytest.raises(TypeError):
            sec.sign_envelope("not a dict")  # type: ignore[arg-type]


# ─── Replay protection ───────────────────────────────────────────────────


class TestReplay:
    def test_nonce_replay_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = sec.sign_envelope({"action": "status"})
        sec.verify_envelope(env)
        with pytest.raises(sec.AuthenticationError, match="replay"):
            sec.verify_envelope(env)

    def test_old_timestamp_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        monkeypatch.setenv("STRANDS_MESH_REPLAY_WINDOW", "5")
        env = sec.sign_envelope({"action": "status"})
        env["ts"] = time.time() - 60.0
        # Re-sign with new ts so we test ts check, not sig check
        # (skip re-signing — just expect AuthenticationError either way)
        with pytest.raises(sec.AuthenticationError):
            sec.verify_envelope(env)

    def test_replay_protection_in_permissive_mode(self):
        env = sec.sign_envelope({"action": "status"})
        sec.verify_envelope(env)
        with pytest.raises(sec.AuthenticationError, match="replay"):
            sec.verify_envelope(env)


# ─── policy_host allowlist ──────────────────────────────────────────────


class TestPolicyHost:
    def test_loopback_allowed_by_default(self):
        assert sec.is_safe_policy_host("localhost")
        assert sec.is_safe_policy_host("127.0.0.1")
        assert sec.is_safe_policy_host("::1")

    def test_arbitrary_host_rejected(self):
        assert not sec.is_safe_policy_host("evil.example.com")
        assert not sec.is_safe_policy_host("8.8.8.8")

    def test_operator_extension_via_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "trusted.example.com,10.0.0.0/24")
        assert sec.is_safe_policy_host("trusted.example.com")
        assert sec.is_safe_policy_host("10.0.0.5")
        assert not sec.is_safe_policy_host("10.0.1.5")
        assert not sec.is_safe_policy_host("untrusted.example.com")

    def test_empty_or_non_string_rejected(self):
        assert not sec.is_safe_policy_host("")
        assert not sec.is_safe_policy_host(None)  # type: ignore[arg-type]


# ─── validate_command ───────────────────────────────────────────────────


class TestValidateCommand:
    def test_status_passes(self):
        out = sec.validate_command({"action": "status"})
        assert out["action"] == "status"

    def test_unknown_action_rejected(self):
        with pytest.raises(sec.ValidationError, match="unknown action"):
            sec.validate_command({"action": "rm_rf"})

    def test_non_dict_rejected(self):
        with pytest.raises(sec.ValidationError):
            sec.validate_command("hello")  # type: ignore[arg-type]

    def test_action_not_string_rejected(self):
        with pytest.raises(sec.ValidationError):
            sec.validate_command({"action": 123})  # type: ignore[dict-item]

    def test_execute_requires_instruction(self):
        with pytest.raises(sec.ValidationError, match="instruction"):
            sec.validate_command({"action": "execute"})

    def test_execute_blank_instruction_rejected(self):
        with pytest.raises(sec.ValidationError, match="instruction"):
            sec.validate_command({"action": "execute", "instruction": "   "})

    def test_execute_long_instruction_rejected(self):
        with pytest.raises(sec.ValidationError, match="exceeds"):
            sec.validate_command({"action": "execute", "instruction": "x" * (sec.MAX_INSTRUCTION_LEN + 1)})

    def test_execute_attacker_policy_host_rejected(self):
        with pytest.raises(sec.ValidationError, match="policy_host"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "pick up cube",
                    "policy_host": "evil.example.com",
                }
            )

    def test_execute_default_localhost_passes(self):
        out = sec.validate_command({"action": "execute", "instruction": "pick up cube"})
        assert out["policy_host"] == "localhost"
        assert out["duration"] == 30.0

    def test_execute_duration_capped(self):
        with pytest.raises(sec.ValidationError, match="duration"):
            sec.validate_command({"action": "execute", "instruction": "go", "duration": sec.MAX_DURATION_S + 1})

    def test_execute_duration_negative_rejected(self):
        with pytest.raises(sec.ValidationError, match="duration"):
            sec.validate_command({"action": "execute", "instruction": "go", "duration": -1})

    def test_execute_duration_non_numeric_rejected(self):
        with pytest.raises(sec.ValidationError, match="duration"):
            sec.validate_command({"action": "execute", "instruction": "go", "duration": "soon"})

    def test_execute_policy_port_bounds(self):
        with pytest.raises(sec.ValidationError, match="policy_port"):
            sec.validate_command({"action": "execute", "instruction": "go", "policy_port": 70_000})
        # valid
        out = sec.validate_command({"action": "execute", "instruction": "go", "policy_port": 5555})
        assert out["policy_port"] == 5555

    def test_step_steps_bounds(self):
        out = sec.validate_command({"action": "step", "steps": 5})
        assert out["steps"] == 5
        with pytest.raises(sec.ValidationError):
            sec.validate_command({"action": "step", "steps": 0})
        with pytest.raises(sec.ValidationError):
            sec.validate_command({"action": "step", "steps": 10_001})

    def test_teleop_receive_requires_source(self):
        with pytest.raises(sec.ValidationError, match="source_peer_id"):
            sec.validate_command({"action": "teleop_receive"})
        out = sec.validate_command({"action": "teleop_receive", "source_peer_id": "leader-1"})
        assert out["source_peer_id"] == "leader-1"

    def test_resume_action_allowed(self):
        out = sec.validate_command({"action": "resume"})
        assert out["action"] == "resume"

    def test_default_action_is_status(self):
        out = sec.validate_command({})
        assert out["action"] == "status"


# ─── per-sender token bucket ────────────────────────────────────────────


class TestPeerRateLimit:
    def test_bucket_allows_initial_burst(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "5/60")
        sec.reset_peer_rate_limits()
        for _ in range(5):
            assert sec.consume_peer_token("alice") is True

    def test_bucket_rejects_after_burst(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "3/60")
        sec.reset_peer_rate_limits()
        for _ in range(3):
            assert sec.consume_peer_token("alice") is True
        assert sec.consume_peer_token("alice") is False

    def test_distinct_senders_have_distinct_buckets(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "2/60")
        sec.reset_peer_rate_limits()
        assert sec.consume_peer_token("alice") is True
        assert sec.consume_peer_token("alice") is True
        assert sec.consume_peer_token("alice") is False
        assert sec.consume_peer_token("bob") is True

    def test_bucket_refills_over_time(self, monkeypatch):
        # rate 10/0.05 → 200 tokens/sec; pause 0.05s and expect ~10 fresh tokens.
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "10/0.5")
        sec.reset_peer_rate_limits()
        for _ in range(10):
            assert sec.consume_peer_token("alice") is True
        assert sec.consume_peer_token("alice") is False
        time.sleep(0.6)
        # After at least one window worth of time, bucket refilled.
        assert sec.consume_peer_token("alice") is True

    def test_anonymous_sender_gets_a_bucket(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "3/60")
        sec.reset_peer_rate_limits()
        for _ in range(3):
            assert sec.consume_peer_token("") is True
        assert sec.consume_peer_token("") is False


class TestTokenBucket:
    def test_tokens_clamp_to_capacity(self):
        b = sec.TokenBucket(capacity=5, rate_per_s=100)
        # Even after long sleep, tokens never exceed capacity.
        time.sleep(0.05)
        # Force an internal recompute by consuming.
        for _ in range(5):
            assert b.consume() is True
        assert b.consume() is False  # capped at 5 in the burst

    def test_consume_zero_always_true(self):
        b = sec.TokenBucket(capacity=1, rate_per_s=0)
        assert b.consume(0) is True


# ─── Coverage for defensive guards & edge cases ──────────────────────────


class TestEnvelopeMalformed:
    def test_missing_ts_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = {"v": 1, "nonce": "abcdef0123456789", "payload": {}}
        with pytest.raises(sec.AuthenticationError, match="ts"):
            sec.verify_envelope(env)

    def test_ts_wrong_type_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = {"v": 1, "ts": "now", "nonce": "abcdef0123456789", "payload": {}}
        with pytest.raises(sec.AuthenticationError, match="ts"):
            sec.verify_envelope(env)

    def test_nonce_too_short_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = {"v": 1, "ts": time.time(), "nonce": "x", "payload": {}}
        with pytest.raises(sec.AuthenticationError, match="nonce"):
            sec.verify_envelope(env)

    def test_payload_not_dict_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        # Build a signed envelope whose payload is a list (still json-valid).
        bad = sec.sign_envelope({"x": 1})
        bad["payload"] = ["not", "a", "dict"]
        # Re-sign so the signature matches the tampered shape — this isolates
        # the "payload not dict" check from the HMAC check.
        body = sec._canonical_bytes({k: bad[k] for k in ("v", "ts", "nonce", "payload")})
        bad["sig"] = sec._hmac_hex(b"k", body)
        sec.clear_replay_cache()
        with pytest.raises(sec.AuthenticationError, match="payload not a dict"):
            sec.verify_envelope(bad)

    def test_sig_missing_in_signed_mode(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = {"v": 1, "ts": time.time(), "nonce": "abcdef0123456789", "payload": {}}
        with pytest.raises(sec.AuthenticationError, match="sig"):
            sec.verify_envelope(env)


class TestPolicyHostCIDR:
    def test_cidr_with_invalid_ip_input(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "10.0.0.0/24")
        # Hostname (not an IP) should reject the CIDR check.
        assert sec.is_safe_policy_host("not.an.ip.address") is False

    def test_cidr_entry_invalid_skipped(self, monkeypatch):
        # Garbage CIDR entries don't crash the matcher.
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "garbage,10.0.0.0/24")
        assert sec.is_safe_policy_host("10.0.0.5") is True


class TestRateLimitEdgeCases:
    def test_invalid_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "garbage-not-a-rate")
        sec.reset_peer_rate_limits()
        # Default = 20/60 — first 20 calls succeed.
        for _ in range(20):
            assert sec.consume_peer_token("x") is True
        assert sec.consume_peer_token("x") is False

    def test_zero_count_clamped_up(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "0/60")
        sec.reset_peer_rate_limits()
        # max(1, 0) → 1 call permitted.
        assert sec.consume_peer_token("y") is True
        assert sec.consume_peer_token("y") is False

    def test_replay_window_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_REPLAY_WINDOW", "xyz")
        # Should not crash; falls back to 60.
        assert sec._replay_window_s() == 60.0


# ─── Regressions for previously-found weaknesses ───────────────────────


class TestForwardSkewBound:
    """Future-stamped envelopes must not pass simply because abs(now-ts) is
    within the window — that would let an attacker borrow forward time and
    replay later when the nonce ages out."""

    def test_envelope_more_than_5s_in_future_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = sec.sign_envelope({"action": "status"})
        env["ts"] = time.time() + 60.0  # 1 minute in the future
        # Re-sign so we test the freshness check, not HMAC.
        body = sec._canonical_bytes({k: env[k] for k in ("v", "ts", "nonce", "payload")})
        env["sig"] = sec._hmac_hex(b"k", body)
        with pytest.raises(sec.AuthenticationError, match="future"):
            sec.verify_envelope(env)

    def test_envelope_within_forward_skew_passes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = sec.sign_envelope({"action": "status"})
        env["ts"] = time.time() + 2.0  # 2s — within MAX_FORWARD_SKEW
        body = sec._canonical_bytes({k: env[k] for k in ("v", "ts", "nonce", "payload")})
        env["sig"] = sec._hmac_hex(b"k", body)
        sec.clear_replay_cache()
        unwrapped = sec.verify_envelope(env)
        assert unwrapped["action"] == "status"


class TestEnvCaps:
    def test_replay_window_capped(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_REPLAY_WINDOW", "999999")
        # Operator misconfig — must clamp.
        assert sec._replay_window_s() == sec._MAX_REPLAY_WINDOW_S

    def test_peer_rate_burst_capped(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "999999/0.1")
        # Burst must not exceed _MAX_PEER_RATE_BURST.
        sec.reset_peer_rate_limits()
        burst, _ = sec._peer_rate_config()
        assert burst <= sec._MAX_PEER_RATE_BURST


class TestNonceLength:
    def test_short_nonce_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PSK", "k")
        env = {
            "v": 1,
            "ts": time.time(),
            "nonce": "tooshort",  # 8 chars — was minimum, now bumped to 16
            "payload": {},
        }
        body = sec._canonical_bytes({k: env[k] for k in ("v", "ts", "nonce", "payload")})
        env["sig"] = sec._hmac_hex(b"k", body)
        with pytest.raises(sec.AuthenticationError, match="nonce"):
            sec.verify_envelope(env)
