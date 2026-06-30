"""Pin: the audit-log writer refuses a symlinked log and fails soft.

``log_safety_event`` is the security-critical writer for the mesh safety
audit log. Its sequence-counter sidecar sibling (``_persist_seq_counters`` /
``_load_seq_counters``) already has symmetric symlink-refusal and fail-soft
coverage (see ``test_audit_seq_persist_symlink.py`` and
``test_audit_seq_symlink.py``), but the audit-LOG write path itself was only
exercised on its happy path. These tests pin its documented contracts
directly:

* a symlink planted at the audit-log path must NOT be followed -- the write
  fails soft (warns, leaves the attacker-chosen target untouched) instead of
  redirecting safety records to attacker-controlled territory. This is the
  symlink-swap defence in ``_ensure_paths`` (the static ``is_symlink`` check
  plus ``O_NOFOLLOW`` on the create);
* any ``OSError`` while opening/writing the log degrades to a WARNING and the
  open file descriptor is closed -- ``log_safety_event`` never raises into the
  safety code path that called it (its module-documented fail-soft contract);
* an ``fsync`` failure is best-effort: the record is still written and the
  call does not crash.
"""

from __future__ import annotations

import json
import os

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolate_audit_state(tmp_path, monkeypatch):
    """Each test starts with a fresh audit dir + reset module state.

    ``audit_log_seeded`` is pinned True so ``_next_seq`` does not walk the
    audit log to seed counters -- these tests target the WRITE path, not the
    seed-from-log fallback, and a symlinked log must not be read either.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = True
    audit._SEQ_COUNTERS.clear()
    yield
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()


def test_log_event_writes_real_private_record(tmp_path) -> None:
    """Happy path: a normal event writes a real (non-symlink) 0o600 log line.

    The positive control that the symlink/fail-soft defences below do not
    break ordinary operation.
    """
    audit.log_safety_event("emergency_stop", "peerA", {"reason": "test"})

    log_path = audit.audit_log_path()
    assert log_path.exists()
    assert not log_path.is_symlink()
    if os.name == "posix":
        assert (log_path.stat().st_mode & 0o777) == 0o600
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert any(r["event"] == "emergency_stop" and r["peer_id"] == "peerA" for r in records)


@pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt",
    reason="symlink semantics differ on Windows; O_NOFOLLOW is 0 there",
)
def test_log_event_refuses_symlinked_log(tmp_path, caplog) -> None:
    """A symlink at the audit-log path must not be followed when writing.

    Pre-fix of the symlink defence, the writer would open the symlink and
    append safety records into the attacker-chosen target. The ``is_symlink``
    refusal in ``_ensure_paths`` fails soft instead: the attacker target is
    left byte-for-byte intact, a WARNING surfaces, and the call does not raise.
    """
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    attacker_target = attacker_dir / "evil.jsonl"
    attacker_target.write_text("ORIGINAL", encoding="utf-8")

    log_path = audit.audit_log_path()
    os.symlink(attacker_target, log_path)

    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit.log_safety_event("emergency_stop", "peerB", {"reason": "test"})  # must not raise

    # The attacker's target file must be left byte-for-byte intact.
    assert attacker_target.read_text(encoding="utf-8") == "ORIGINAL"
    # The symlink itself must still be a symlink (not replaced by a real file).
    assert log_path.is_symlink()
    # A WARNING must surface so operators can attribute the dropped record.
    assert any("SYMLINK" in rec.message or "symlink" in rec.message.lower() for rec in caplog.records), (
        "expected a WARNING about the symlinked audit log"
    )


def test_log_event_fails_soft_when_open_raises(tmp_path, monkeypatch, caplog) -> None:
    """An OSError while opening the log degrades to a WARNING, never raises.

    Audit-log write failures must never propagate into the safety code path
    that called ``log_safety_event`` (module-documented fail-soft contract).
    """

    real_open = audit.os.open

    def _boom(path, flags, *args, **kwargs):
        # Fail only the append-write open of the audit log; let every other
        # os.open (seq sidecar, lockfile, the O_CREAT|O_EXCL create) succeed.
        if flags & os.O_APPEND:
            raise OSError("disk full")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", _boom)

    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit.log_safety_event("emergency_stop", "peerC", {"reason": "test"})  # must not raise

    assert any("failed to write" in rec.message for rec in caplog.records), (
        "expected a fail-soft WARNING when the audit-log write hits an OSError"
    )


def test_log_event_records_survive_fsync_failure(tmp_path, monkeypatch) -> None:
    """An ``fsync`` failure is best-effort: the record is still written.

    On filesystems that reject ``fsync`` the write loses durability, not the
    record -- the data is already in the page cache and the call must not crash.
    """

    def _no_fsync(_fd):
        raise OSError("fsync not supported")

    monkeypatch.setattr(audit.os, "fsync", _no_fsync)

    audit.log_safety_event("emergency_stop", "peerD", {"reason": "test"})  # must not raise

    log_path = audit.audit_log_path()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert any(r["event"] == "emergency_stop" and r["peer_id"] == "peerD" for r in records)


def test_log_event_closes_fd_when_fdopen_raises(tmp_path, monkeypatch, caplog) -> None:
    """If ``fdopen``/write raises after the fd is opened, the fd is closed.

    The append open succeeds but wrapping it in a buffered writer fails; the
    cleanup path must close the raw fd (no descriptor leak) and the OSError
    must still fail soft into a WARNING rather than propagate.
    """
    closed: list[int] = []
    real_close = audit.os.close

    def _tracking_close(fd):
        closed.append(fd)
        return real_close(fd)

    def _boom_fdopen(_fd, *_args, **_kwargs):
        raise OSError("cannot wrap fd")

    monkeypatch.setattr(audit.os, "fdopen", _boom_fdopen)
    monkeypatch.setattr(audit.os, "close", _tracking_close)

    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit.log_safety_event("emergency_stop", "peerE", {"reason": "test"})  # must not raise

    # The raw fd opened for the append write was closed by the cleanup path.
    assert closed, "expected the append-write fd to be closed after fdopen failed"
    assert any("failed to write" in rec.message for rec in caplog.records), (
        "expected a fail-soft WARNING when fdopen raises"
    )
