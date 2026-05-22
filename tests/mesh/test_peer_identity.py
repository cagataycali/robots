"""Adversarial tests for per-peer cryptographic identity (R9).

These are NOT inspection / surface tests — every test in this file
simulates a concrete attacker behaviour against the verifier and asserts
the rejection. They are the empirical complement to the existing
verified-by-inspection coverage and answer the reviewer's question about
"hangi vector'ün gerçek adversarial testi var".

Threat model summary
--------------------
The fleet PSK ``STRANDS_MESH_PSK`` proves *fleet membership*. An insider
who holds the PSK (e.g. a compromised camera in tier A2 of the threat
matrix) can mint a valid HMAC for ANY ``sender_id`` value. R9 adds a
per-peer HMAC key on top so ``sender_id`` is bound cryptographically to
the holder of that peer's key.

Scope of these tests
--------------------
* **Vector #9** (presence spoofing) — Mallory cannot pin a different
  key for an already-bound peer, and cannot mint id_sigs for a peer
  whose key she does not hold.
* **Vector #10** (per-sender rate limit / audit attribution forgery) —
  Mallory cannot put another peer's ``sender_id`` in the payload
  without a matching kid + id_sig.
* **Strict mode** (``STRANDS_MESH_REQUIRE_PEER_IDENTITY=true``) — every
  envelope without a kid is rejected.
* **TOFU race window** — first-pinned-wins is enforced; subsequent
  conflicts are surfaced as :class:`AuthenticationError` so the audit
  log can record the rejection.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import identity
from strands_robots.mesh import security as sec

# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_identity(monkeypatch, tmp_path):
    """Each test runs with an empty directory and clean nonce cache."""
    monkeypatch.setenv("STRANDS_MESH_PEER_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.setenv("STRANDS_MESH_PSK", "test-fleet-psk-not-secret")
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_PEER_IDENTITY", raising=False)
    monkeypatch.delenv("STRANDS_MESH_PEER_IDENTITY", raising=False)
    monkeypatch.delenv("STRANDS_MESH_PEER_KEY", raising=False)
    monkeypatch.delenv("STRANDS_MESH_PEER_KEY_FILE", raising=False)
    sec._PROCESS_STATE.psk_warned = False
    sec.clear_replay_cache()
    identity.reset_directory()
    yield
    sec.clear_replay_cache()
    identity.reset_directory()


@pytest.fixture
def alice_key():
    return identity.configure_local_peer("alice")


@pytest.fixture
def bob_key():
    return identity.configure_local_peer("bob")


@pytest.fixture
def mallory_key():
    """Mallory is an authenticated insider — has PSK, has her own peer
    key, but does NOT have alice's per-peer key."""
    return identity.configure_local_peer("mallory")


# ─── Per-peer key generation ────────────────────────────────────────────


class TestPerPeerKey:
    """The local key store is the foundation of identity."""

    def test_keys_are_unique_per_peer(self):
        a = identity.configure_local_peer("alice")
        b = identity.configure_local_peer("bob")
        assert len(a) == identity.PEER_KEY_LEN
        assert len(b) == identity.PEER_KEY_LEN
        assert a != b

    def test_key_is_persistent_across_calls(self):
        a1 = identity.configure_local_peer("alice")
        a2 = identity.configure_local_peer("alice")
        assert a1 == a2

    def test_invalid_peer_id_disables_identity(self, caplog):
        # Path-traversal style id MUST not become a key path.
        key = identity.configure_local_peer("../etc/shadow")
        assert key is None

    def test_opt_out_via_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_IDENTITY", "false")
        assert identity.configure_local_peer("alice") is None

    def test_env_override_hex(self, monkeypatch):
        # 32 zero bytes hex
        monkeypatch.setenv("STRANDS_MESH_PEER_KEY", "00" * identity.PEER_KEY_LEN)
        key = identity.configure_local_peer("alice")
        assert key == bytes(identity.PEER_KEY_LEN)

    def test_env_override_bad_hex_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_KEY", "nothex")
        with pytest.raises(identity.IdentityError):
            identity.configure_local_peer("alice")

    def test_env_override_wrong_length_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_KEY", "00" * 8)  # too short
        with pytest.raises(identity.IdentityError):
            identity.configure_local_peer("alice")

    def test_key_file_mode_is_restrictive(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_PEER_KEY_DIR", str(tmp_path))
        identity.configure_local_peer("alice")
        path = tmp_path / "alice.key"
        assert path.exists()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"


# ─── Directory pinning semantics ────────────────────────────────────────


class TestDirectoryPin:
    def test_pin_then_lookup(self, alice_key):
        assert sec.pin_peer_identity("alice", alice_key) is True
        assert identity.get_directory().lookup("alice") == alice_key

    def test_repin_same_key_is_idempotent(self, alice_key):
        sec.pin_peer_identity("alice", alice_key)
        assert sec.pin_peer_identity("alice", alice_key) is False

    def test_repin_different_key_raises(self, alice_key, mallory_key):
        sec.pin_peer_identity("alice", alice_key)
        with pytest.raises(sec.AuthenticationError, match="already bound"):
            sec.pin_peer_identity("alice", mallory_key)

    def test_drop_then_repin(self, alice_key, mallory_key):
        sec.pin_peer_identity("alice", alice_key)
        assert sec.drop_peer_identity("alice") is True
        # After drop, a NEW key for the same peer can be pinned (operator
        # rotation flow).
        assert sec.pin_peer_identity("alice", mallory_key) is True

    def test_invalid_peer_id_rejected(self, alice_key):
        with pytest.raises(identity.IdentityError):
            identity.get_directory().pin("../etc/shadow", alice_key)
        with pytest.raises(identity.IdentityError):
            identity.get_directory().pin("alice", b"too-short")

    def test_load_directory_from_dir(self, tmp_path, alice_key, bob_key):
        # Operator pre-distribution mode.
        (tmp_path / "alice.key").write_bytes(alice_key)
        (tmp_path / "bob.key").write_bytes(bob_key)
        n = identity.load_peer_directory_from_dir(tmp_path)
        assert n == 2
        assert sorted(sec.known_peer_identities()) == ["alice", "bob"]


# ─── Adversarial: Vector #9 (presence/sender spoofing) ──────────────────


class TestSenderIdSpoof:
    """Authenticated insider Mallory has the PSK + her own key. Can she
    publish messages claiming to be alice? Each test below is a real
    on-the-wire attack."""

    def test_mallory_signs_with_own_key_but_claims_kid_alice(self, alice_key, bob_key, mallory_key):
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("bob", bob_key)

        # Mallory builds an envelope claiming to be alice but signs with
        # HER key. Her id_sig will not verify under alice's pinned key.
        env = sec.sign_envelope(
            {"sender_id": "alice", "command": {"action": "status"}},
            peer_id="alice",
            peer_key=mallory_key,
        )
        with pytest.raises(sec.AuthenticationError, match="identity signature mismatch"):
            sec.verify_envelope(env, scope="bob")

    def test_psk_only_envelope_with_kid_alice_requires_id_sig(self, alice_key, bob_key):
        """An attacker with PSK alone cannot fake alice once alice is pinned —
        the envelope MUST carry a valid id_sig under alice's key."""
        sec.pin_peer_identity("alice", alice_key)

        # Mallory has PSK, no per-peer key for alice. She crafts an envelope
        # with kid='alice' and a valid PSK sig but no id_sig.
        import hashlib
        import hmac
        import json
        import time as _time
        import uuid

        ts = _time.time()
        env = {
            "v": 1,
            "ts": ts,
            "nonce": uuid.uuid4().hex,
            "kid": "alice",
            "payload": {"sender_id": "alice", "command": {"action": "stop"}},
        }
        body = json.dumps(env, sort_keys=True, separators=(",", ":")).encode()
        env["sig"] = hmac.new(b"test-fleet-psk-not-secret", body, hashlib.sha256).hexdigest()

        with pytest.raises(sec.AuthenticationError, match="missing id_sig"):
            sec.verify_envelope(env, scope="bob")

    def test_psk_only_envelope_with_unbound_sender_id_rejected(self, alice_key, bob_key):
        """The residual surface: insider drops the kid entirely and stuffs
        sender_id into the payload. Closed by the sender-id binding rule."""
        sec.pin_peer_identity("alice", alice_key)

        # PSK-only signature, no kid, sender_id='alice' in payload.
        env = sec.sign_envelope({"sender_id": "alice", "command": {"action": "status"}})
        with pytest.raises(sec.AuthenticationError, match="payload.sender_id.*envelope kid is missing"):
            sec.verify_envelope(env, scope="bob")

    def test_kid_mismatch_with_payload_sender_id(self, alice_key, mallory_key):
        """Envelope kid says one peer, payload says another. Reject."""
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("mallory", mallory_key)

        # Mallory signs honestly as herself but lies in the payload.
        env = sec.sign_envelope(
            {"sender_id": "alice", "command": {"action": "stop"}},
            peer_id="mallory",
            peer_key=mallory_key,
        )
        with pytest.raises(sec.AuthenticationError, match="disagrees with payload.sender_id"):
            sec.verify_envelope(env, scope="bob")

    def test_legitimate_signed_message_passes(self, alice_key, bob_key):
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("bob", bob_key)

        env = sec.sign_envelope(
            {"sender_id": "alice", "command": {"action": "status"}},
            peer_id="alice",
            peer_key=alice_key,
        )
        payload = sec.verify_envelope(env, scope="bob")
        assert payload["sender_id"] == "alice"

    def test_tampering_with_kid_breaks_id_sig(self, alice_key, bob_key, mallory_key):
        """An on-the-wire MITM rewrites the kid field — id_sig recomputation
        catches it."""
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("mallory", mallory_key)

        env = sec.sign_envelope(
            {"sender_id": "alice", "command": {"action": "status"}},
            peer_id="alice",
            peer_key=alice_key,
        )
        # MITM: change kid (and re-sign with PSK so fleet sig still passes).
        import hashlib
        import hmac
        import json

        env["kid"] = "mallory"
        body = json.dumps(
            {k: env[k] for k in ("v", "ts", "nonce", "kid", "payload")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        env["sig"] = hmac.new(b"test-fleet-psk-not-secret", body, hashlib.sha256).hexdigest()
        # Note: id_sig is now stale because we changed the kid in the
        # canonical body.

        with pytest.raises(sec.AuthenticationError, match="identity signature mismatch"):
            sec.verify_envelope(env, scope="bob")


# ─── Adversarial: Vector #10 (rate-limit / audit attribution forgery) ───


class TestRateLimitAttribution:
    """A legacy version of the verifier accepted ``payload.sender_id``
    at face value. Mallory could burn alice's per-sender bucket. With
    the kid binding, she cannot."""

    def test_mallory_cannot_forge_alices_rate_bucket(self, alice_key, mallory_key):
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("mallory", mallory_key)
        sec.reset_peer_rate_limits()

        # Mallory floods commands claiming sender_id='alice'. Each one is
        # rejected at the verifier — she cannot reach consume_peer_token
        # at all under alice's name.
        rejections = 0
        for _ in range(50):
            sec.clear_replay_cache()  # fresh nonce per attempt
            env = sec.sign_envelope(
                {"sender_id": "alice", "command": {"action": "status"}},
                peer_id="alice",
                peer_key=mallory_key,  # signs with HER key, not alice's
            )
            try:
                sec.verify_envelope(env, scope="bob")
            except sec.AuthenticationError:
                rejections += 1
        assert rejections == 50

        # And alice's bucket is still full: alice can still send legitimate
        # commands at full burst. (Indirect proof: token consumption works.)
        for _ in range(20):  # default burst is 20
            assert sec.consume_peer_token("alice") is True


# ─── Strict identity mode ───────────────────────────────────────────────


class TestStrictIdentityMode:
    def test_envelope_without_kid_rejected_in_strict_mode(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_REQUIRE_PEER_IDENTITY", "true")

        env = sec.sign_envelope({"action": "status"})
        with pytest.raises(sec.AuthenticationError, match="missing kid"):
            sec.verify_envelope(env, scope="bob")

    def test_envelope_with_kid_no_id_sig_rejected_in_strict_mode(self, monkeypatch, alice_key):
        monkeypatch.setenv("STRANDS_MESH_REQUIRE_PEER_IDENTITY", "true")
        # No pinned key for alice yet — PSK-only verifier would normally
        # accept this in non-strict mode. Strict mode requires id_sig.
        env = sec.sign_envelope(
            {"action": "status"},
            peer_id="alice",
            peer_key=None,  # explicitly omit id_sig
        )
        assert "id_sig" not in env
        with pytest.raises(sec.AuthenticationError, match="missing id_sig"):
            sec.verify_envelope(env, scope="bob")

    def test_strict_mode_passes_with_full_identity(self, monkeypatch, alice_key, bob_key):
        monkeypatch.setenv("STRANDS_MESH_REQUIRE_PEER_IDENTITY", "true")
        sec.pin_peer_identity("alice", alice_key)

        env = sec.sign_envelope(
            {"action": "status"},
            peer_id="alice",
            peer_key=alice_key,
        )
        payload = sec.verify_envelope(env, scope="bob")
        assert payload == {"action": "status"}


# ─── TOFU bootstrap behaviour ───────────────────────────────────────────


class TestTOFUBootstrap:
    """First valid presence pins; subsequent conflicts surface."""

    def test_first_psk_presence_can_pin_unknown_peer(self, alice_key, bob_key):
        # alice has not been pinned yet; her PSK-signed envelope passes
        # verify_envelope (kid present, no pinned key, no id_sig required)
        # and the receiving Mesh's _on_presence calls pin_peer_identity.
        sec.pin_peer_identity("bob", bob_key)
        env = sec.sign_envelope(
            {"robot_id": "alice", "robot_type": "robot", "peer_key": alice_key.hex()},
            peer_id="alice",
            peer_key=alice_key,
        )
        # The verifier accepts (alice unpinned -> id_sig optional).
        payload = sec.verify_envelope(env, scope="bob")
        assert payload["robot_id"] == "alice"
        # Application code (in core.py::_on_presence) would now call:
        sec.pin_peer_identity("alice", alice_key)
        # And subsequent envelopes from a non-alice key under kid=alice
        # are rejected.

    def test_second_attempt_with_different_key_after_pin_fails(self, alice_key, mallory_key, bob_key):
        """Once alice is pinned, mallory cannot rebind alice's id."""
        sec.pin_peer_identity("alice", alice_key)
        sec.pin_peer_identity("bob", bob_key)

        # Mallory presses her own key into a presence claiming kid=alice.
        # The envelope itself is rejected at verify time (id_sig mismatch),
        # so the application-layer pin_peer_identity is never reached.
        env = sec.sign_envelope(
            {"robot_id": "alice", "robot_type": "robot", "peer_key": mallory_key.hex()},
            peer_id="alice",
            peer_key=mallory_key,
        )
        with pytest.raises(sec.AuthenticationError, match="identity signature mismatch"):
            sec.verify_envelope(env, scope="bob")

    def test_silent_rotation_blocked_at_directory_level(self, alice_key, mallory_key):
        """Even if an attacker bypasses verify (e.g. drops the verifier),
        the pin_peer_identity entry point itself refuses silent rotation."""
        sec.pin_peer_identity("alice", alice_key)
        with pytest.raises(sec.AuthenticationError, match="already bound"):
            sec.pin_peer_identity("alice", mallory_key)


# ─── Backward compatibility ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """Existing peers without per-peer keys must still interoperate
    in non-strict mode. This is the migration surface."""

    def test_unsigned_envelope_with_no_kid_still_accepted(self):
        env = sec.sign_envelope({"action": "status"})
        payload = sec.verify_envelope(env, scope="bob")
        assert payload == {"action": "status"}

    def test_envelope_with_kid_but_no_id_sig_accepted_when_unpinned(self, alice_key):
        # alice not yet pinned -> bootstrap path; PSK alone suffices.
        env = sec.sign_envelope({"action": "status"}, peer_id="alice", peer_key=None)
        assert "id_sig" not in env
        payload = sec.verify_envelope(env, scope="bob")
        assert payload == {"action": "status"}

    def test_legacy_bare_payload_still_passes_in_permissive(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
        monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
        sec._PROCESS_STATE.psk_warned = False

        legacy = {"action": "status"}
        out = sec.verify_envelope(legacy, scope="bob")
        assert out == legacy
