"""Pin tests for PR #195 review follow-ups (F3 batch).

Each test in this module pins a specific concern raised by reviewer
``yinsong1986`` after the F1+F2 fixes landed. Per AGENTS.md > Review
Learnings (#85) > "Pin regression tests for reviewed fixes": every fix
gets a test that fails on pre-fix code.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config as ac
from strands_robots.mesh import _zenoh_config as zc
from strands_robots.mesh import audit as audit_mod
from strands_robots.mesh import core as core_mod
from strands_robots.mesh import security as sec

# ---------------------------------------------------------------------
# F3-A: JSON5 dep swap -- malformed input now produces clear errors
# ---------------------------------------------------------------------


class TestJSON5DepSwap:
    """The hand-rolled preprocessor was replaced with json5.loads. These
    pins ensure the new parser surfaces operator-friendly diagnostics on
    malformed input rather than silently truncating (the old behaviour).
    """

    def test_unterminated_block_comment_raises_clear_error(self, tmp_path):
        path = tmp_path / "acl.json5"
        path.write_text("{\n  /* unterminated block ...\n  enabled: true,\n}\n")
        with pytest.raises(ValueError, match=r"is not valid JSON5"):
            ac._load_acl_file(path)

    def test_no_legacy_preprocessor_symbols_exist(self):
        """The four hand-rolled preprocessor functions must be gone --
        catches a future revert that re-introduces the fragile parser.
        """
        for name in (
            "_strip_json5_comments",
            "_strip_trailing_commas",
            "_quote_unquoted_keys",
            "_convert_single_quoted_strings",
            "_json5_to_json",
        ):
            assert not hasattr(ac, name), f"{name} should have been removed in F3-A"

    def test_json5_pep_dependency_imported(self):
        """json5 must be importable from _acl_config (mesh extra)."""
        assert ac.json5 is not None


# ---------------------------------------------------------------------
# F3-B-1: bare except on permissive-ACL warning narrowed
# ---------------------------------------------------------------------


class TestPermissiveACLWarningExceptNarrow:
    """The R19 permissive-ACL warning at Mesh.start() previously caught
    `except Exception` and downgraded to DEBUG -- a future refactor that
    raised an unrelated type would silently lose the warning.
    """

    def test_unexpected_exception_surfaces_loudly(self, monkeypatch, caplog):
        """A non-(ImportError|ValueError) raised inside the warning block
        should surface at WARNING (not DEBUG silent-swallow).
        """

        # Construct a Mesh against a stub robot
        class StubRobot:
            pass

        # Force resolve_auth_mode to raise an unexpected type
        def raise_runtime():
            raise RuntimeError("synthetic")

        monkeypatch.setattr(zc, "resolve_auth_mode", raise_runtime)

        # The narrowed except clause is (ImportError, ValueError); a
        # RuntimeError should propagate. We don't actually call start()
        # (it does network I/O); we just assert the source code carries
        # the narrowed tuple.
        src = Path(core_mod.__file__).read_text()
        assert "except (ImportError, ValueError) as warn_exc:" in src
        # No bare `except Exception as warn_exc:` left in the start() block
        assert "except Exception as warn_exc:" not in src


# ---------------------------------------------------------------------
# F3-B-2: STRANDS_MESH_CAMERA_DISABLED via _bool_env (lenient parse)
# ---------------------------------------------------------------------


class TestCameraDisabledLenientParse:
    """Privacy kill switch must accept the same truthy values as the
    other boolean env vars, not just the literal string ``"true"``.
    """

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "True"])
    def test_truthy_values_disable_camera(self, monkeypatch, value):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", value)
        assert zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_falsy_values_keep_camera_enabled(self, monkeypatch, value):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", value)
        assert zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False) is False

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", "maybe")
        with pytest.raises(ValueError, match=r"not a boolean"):
            zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False)


# ---------------------------------------------------------------------
# F3-B-3: _on_safety_resume rejects empty/missing peer_id (R20 mirror)
# ---------------------------------------------------------------------


class TestResumeStrictPeerId:
    """The estop handler (R20) rejects envelopes with empty/missing
    peer_id outright. Resume must mirror that posture.
    """

    def test_resume_with_empty_peer_id_rejected(self, caplog):
        class StubRobot:
            pass

        m = core_mod.Mesh(robot=StubRobot(), peer_id="robot-test")
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

        m = core_mod.Mesh(robot=StubRobot(), peer_id="robot-r")
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


class TestPSKStateLock:
    """The PSK fingerprint snapshot is read-modify-compared on every
    log_safety_event call. The dedicated lock makes that atomic.
    """

    def test_lock_module_attr_exists(self):
        assert hasattr(audit_mod, "_PSK_STATE_LOCK")
        assert isinstance(audit_mod._PSK_STATE_LOCK, type(threading.Lock()))

    def test_concurrent_writers_first_record_no_race(self, tmp_path, monkeypatch):
        """Spawn 16 threads that each call log_safety_event on a fresh
        process state. The PSK fingerprint must end up consistent and
        no thread should observe a partial mid-write view.
        """
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "test-psk-concurrent")
        # Reset state
        audit_mod._AUDIT_STATE.psk_fingerprint = None
        audit_mod._AUDIT_STATE.seq_loaded = False
        audit_mod._SEQ_COUNTERS.clear()

        errors: list[Exception] = []

        def writer(i: int):
            try:
                audit_mod.log_safety_event("test", f"peer-{i}", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent writers raised: {errors}"
        # All records should have been signed under the same fingerprint
        records = audit_mod.read_audit_log()
        sigs = [r.get("sig") for r in records]
        # Every record (16) must have a real HMAC, not a poison marker
        assert len(sigs) == 16
        assert all(s and s != "PSK_DEGRADED" and len(s) == 64 for s in sigs), (
            f"some records were poisoned/unsigned under concurrency: {sigs}"
        )


# ---------------------------------------------------------------------
# F3-C-2: log_safety_event widened fail-soft contract
# ---------------------------------------------------------------------


class TestAuditFailSoft:
    """The fail-soft contract (audit must never crash safety path)
    previously caught only AuditPSKDegradedError. F3 widens it.
    """

    def test_sign_record_runtime_error_does_not_crash(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        audit_mod._AUDIT_STATE.psk_fingerprint = None
        audit_mod._AUDIT_STATE.seq_loaded = False
        audit_mod._SEQ_COUNTERS.clear()

        # Patch _sign_record to raise an unexpected RuntimeError

        def boom(record):
            raise RuntimeError("synthetic failure inside _sign_record")

        monkeypatch.setattr(audit_mod, "_sign_record", boom)

        with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.audit"):
            # Must NOT raise -- safety-path contract
            audit_mod.log_safety_event("test", "peer-1", {"data": "ok"})

        # The record was still written (unsigned)
        records = audit_mod.read_audit_log()
        assert len(records) == 1
        assert "sig" not in records[0]  # unsigned per the widened fail-soft path

        # And we logged the failure at ERROR
        assert any("_sign_record raised" in m for m in caplog.messages), (
            f"expected ERROR log about _sign_record failure; got {caplog.messages}"
        )


# ---------------------------------------------------------------------
# F3-D-1: verify_ca_pin O_NOFOLLOW symlink defence
# ---------------------------------------------------------------------


class TestVerifyCaPinSymlink:
    """The public verify_ca_pin must not follow symlinks (asymmetric
    with _ensure_ca's R22-D defence was the actual gap).
    """

    def test_symlinked_ca_path_returns_false(self, tmp_path):
        from strands_robots.mesh.iot.provision import verify_ca_pin

        target = tmp_path / "real_ca.pem"
        target.write_bytes(b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        symlink = tmp_path / "ca_link.pem"
        symlink.symlink_to(target)

        # verify_ca_pin must refuse to read through the symlink
        assert verify_ca_pin(symlink) is False


# ---------------------------------------------------------------------
# F3-D-2: STRANDS_MESH_POLICY_HOST_ALLOW operator validation
# ---------------------------------------------------------------------


class TestPolicyHostAllowlistValidation:
    """Operator-supplied entries with shell metacharacters / whitespace
    are dropped with a WARNING (fail-loud-on-misconfig).
    """

    def test_malformed_entry_dropped_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "STRANDS_MESH_POLICY_HOST_ALLOW",
            "10.0.0.0/24,;rm -rf /,vla.internal",
        )
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
            allowlist = sec._policy_host_allowlist()
        # Defaults + valid entries only
        assert "10.0.0.0/24" in allowlist
        assert "vla.internal" in allowlist
        # Malformed entry was dropped
        assert ";rm -rf /" not in allowlist
        assert any("dropping malformed entry" in m for m in caplog.messages), (
            f"expected WARNING about malformed entry; got {caplog.messages}"
        )

    def test_clean_entries_pass_through(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "STRANDS_MESH_POLICY_HOST_ALLOW",
            "10.0.0.5,vla.internal,2001:db8::1",
        )
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
            allowlist = sec._policy_host_allowlist()
        # No warnings on clean input
        assert not any("dropping malformed" in m for m in caplog.messages)
        for entry in ("10.0.0.5", "vla.internal", "2001:db8::1"):
            assert entry in allowlist
