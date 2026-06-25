"""Audit-log size-based rotation and bound-resolution tests.

Cover the disk-bounding guarantees of :mod:`strands_robots.mesh.audit`:

* :func:`audit._resolve_log_max_bytes` and
  :func:`audit._resolve_log_max_files` clamp operator-supplied env vars
  to safe ranges (reject non-numeric, non-positive, and over-cap values)
  so a typo cannot turn the audit log back into an unbounded growth
  surface or starve rotation.
* :func:`audit._rotate_log_if_needed` is a no-op below the size cap,
  refuses to rotate a symlinked log, and otherwise cascades numbered
  suffixes (``.1`` .. ``.N``) while discarding history past
  ``max_files``.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolated_audit(monkeypatch, tmp_path):
    """Each test gets a fresh audit dir and clean rotation env vars."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_MAX_BYTES", raising=False)
    monkeypatch.delenv("STRANDS_MESH_AUDIT_MAX_FILES", raising=False)
    yield


# --- _resolve_log_max_bytes -------------------------------------------------


def test_max_bytes_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_AUDIT_MAX_BYTES", raising=False)
    assert audit._resolve_log_max_bytes() == audit._DEFAULT_LOG_MAX_BYTES


def test_max_bytes_rejects_non_numeric(monkeypatch, caplog):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "not-a-number")
    with caplog.at_level("WARNING"):
        assert audit._resolve_log_max_bytes() == audit._DEFAULT_LOG_MAX_BYTES
    assert "invalid" in caplog.text


def test_max_bytes_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "0")
    assert audit._resolve_log_max_bytes() == audit._DEFAULT_LOG_MAX_BYTES
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "-5")
    assert audit._resolve_log_max_bytes() == audit._DEFAULT_LOG_MAX_BYTES


def test_max_bytes_clamps_above_hard_cap(monkeypatch, caplog):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", str(audit._LOG_MAX_BYTES_CAP + 1))
    with caplog.at_level("WARNING"):
        assert audit._resolve_log_max_bytes() == audit._LOG_MAX_BYTES_CAP
    assert "exceeds hard cap" in caplog.text


def test_max_bytes_accepts_valid_override(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "4096")
    assert audit._resolve_log_max_bytes() == 4096


# --- _resolve_log_max_files -------------------------------------------------


def test_max_files_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_AUDIT_MAX_FILES", raising=False)
    assert audit._resolve_log_max_files() == audit._DEFAULT_LOG_MAX_FILES


def test_max_files_non_numeric_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "lots")
    assert audit._resolve_log_max_files() == audit._DEFAULT_LOG_MAX_FILES


def test_max_files_floor_is_one(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "0")
    assert audit._resolve_log_max_files() == 1


def test_max_files_clamps_to_hard_cap(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", str(audit._LOG_MAX_FILES_CAP + 50))
    assert audit._resolve_log_max_files() == audit._LOG_MAX_FILES_CAP


def test_max_files_accepts_valid_override(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "3")
    assert audit._resolve_log_max_files() == 3


# --- _rotate_log_if_needed --------------------------------------------------


def test_rotate_noop_below_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "1000")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("small\n", encoding="utf-8")
    audit._rotate_log_if_needed(log, current_size=10)
    # Below the cap: file stays put, no rotation created.
    assert log.exists()
    assert not (tmp_path / "mesh_audit.jsonl.1").exists()


def test_rotate_refuses_symlinked_log(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    real = tmp_path / "real_target.jsonl"
    real.write_text("x" * 100, encoding="utf-8")
    link = tmp_path / "mesh_audit.jsonl"
    link.symlink_to(real)
    with caplog.at_level("WARNING"):
        audit._rotate_log_if_needed(link, current_size=100)
    assert "refusing to rotate symlinked" in caplog.text
    # The link and its target are untouched; no numbered rotation made.
    assert link.is_symlink()
    assert not (tmp_path / "mesh_audit.jsonl.1").exists()


def test_rotate_moves_active_log_to_suffix_one(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "3")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("active-data" * 5, encoding="utf-8")
    audit._rotate_log_if_needed(log, current_size=1000)
    rotated = tmp_path / "mesh_audit.jsonl.1"
    assert rotated.exists()
    assert rotated.read_text(encoding="utf-8").startswith("active-data")
    # The active path is now free for the next write to recreate.
    assert not log.exists()


def test_rotate_cascades_and_discards_past_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "2")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("gen0", encoding="utf-8")
    (tmp_path / "mesh_audit.jsonl.1").write_text("gen1", encoding="utf-8")
    (tmp_path / "mesh_audit.jsonl.2").write_text("gen2", encoding="utf-8")

    audit._rotate_log_if_needed(log, current_size=1000)

    # max_files=2: .2 (oldest) is discarded, .1 -> .2, active -> .1.
    assert (tmp_path / "mesh_audit.jsonl.1").read_text(encoding="utf-8") == "gen0"
    assert (tmp_path / "mesh_audit.jsonl.2").read_text(encoding="utf-8") == "gen1"
    assert not (tmp_path / "mesh_audit.jsonl.3").exists()
    assert not log.exists()


# --- _rotate_log_if_needed: symlink + fail-soft cascade branches -----------


def test_rotate_discards_symlinked_overflow_without_following_it(monkeypatch, tmp_path, caplog):
    """An overflow rotated log that is a SYMLINK is unlinked, never followed.

    When a rotation past ``max_files`` lands on a numbered suffix that an
    attacker has pre-created as a symlink, the cascade deletes the link inode
    -- it must not traverse the link and redirect the delete onto the link
    target (a co-tenant file, ``/dev/null``, etc.). ``Path.unlink`` removes
    the link rather than following it, so the target survives untouched, and
    a forensic WARNING records the discard.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "1")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("gen0", encoding="utf-8")
    # max_files=1: the cascade visits suffix ".1", and since 1 + 1 > 1 that
    # slot is past the cap and gets discarded. Plant ".1" as a symlink to a
    # precious co-tenant file to prove the discard never follows the link.
    precious = tmp_path / "precious_cotenant.txt"
    precious.write_text("do-not-touch", encoding="utf-8")
    overflow_link = tmp_path / "mesh_audit.jsonl.1"
    overflow_link.symlink_to(precious)

    with caplog.at_level("WARNING"):
        audit._rotate_log_if_needed(log, current_size=1000)

    # The discarded symlink was removed and then recreated as a regular file
    # by the active-log rename (active -> .1); it must no longer be a symlink.
    assert not overflow_link.is_symlink()
    # The link target is fully intact -- the discard deleted the link inode,
    # never truncated or redirected onto the target.
    assert precious.exists()
    assert precious.read_text(encoding="utf-8") == "do-not-touch"
    assert "discarding symlinked rotated log" in caplog.text
    # The cascade still completed: active -> .1 (now a regular file).
    assert overflow_link.is_file()
    assert overflow_link.read_text(encoding="utf-8") == "gen0"
    assert not log.exists()


def test_rotate_is_failsoft_when_cascade_replace_raises(monkeypatch, tmp_path, caplog):
    """A mid-cascade ``os.replace`` failure (e.g. EXDEV) is logged, not raised.

    Audit rotation runs on the safety hot path under ``_WRITE_LOCK``; a
    cross-device-rename or permission error while shuffling numbered suffixes
    must degrade to a WARNING rather than propagate and crash the writer.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "3")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("gen0", encoding="utf-8")
    rotated_one = tmp_path / "mesh_audit.jsonl.1"
    rotated_one.write_text("gen1", encoding="utf-8")

    real_replace = audit.os.replace

    def _replace_fails_on_cascade(src, dst):
        # Fail the cascade step (.1 -> .2) but let the final active -> .1
        # rename through so we can prove the function pressed on.
        if str(src).endswith("mesh_audit.jsonl.1"):
            raise OSError("simulated EXDEV during cascade")
        return real_replace(src, dst)

    monkeypatch.setattr(audit.os, "replace", _replace_fails_on_cascade)

    with caplog.at_level("WARNING"):
        # Must not raise.
        audit._rotate_log_if_needed(log, current_size=1000)

    assert "rotation cascade failed" in caplog.text
    # The active log still rotated to .1 despite the cascade hiccup.
    assert rotated_one.read_text(encoding="utf-8") == "gen0"
    assert not log.exists()


def test_rotate_is_failsoft_when_active_rename_raises(monkeypatch, tmp_path, caplog):
    """A failure renaming the active log to ``.1`` is logged, not raised.

    The final ``os.replace(active, active.1)`` is the last step of rotation.
    If it fails (full disk, EACCES) the audit subsystem must stay alive: the
    next write recreates/appends rather than crashing safety.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "10")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "3")
    log = tmp_path / "mesh_audit.jsonl"
    log.write_text("active-bytes" * 4, encoding="utf-8")

    real_replace = audit.os.replace

    def _replace_fails_on_active(src, dst):
        if str(src).endswith("mesh_audit.jsonl"):
            raise OSError("simulated ENOSPC renaming active log")
        return real_replace(src, dst)

    monkeypatch.setattr(audit.os, "replace", _replace_fails_on_active)

    with caplog.at_level("WARNING"):
        audit._rotate_log_if_needed(log, current_size=1000)

    assert "could not rotate" in caplog.text
    # Active log untouched (rename failed) -- no .1 was created.
    assert log.exists()
    assert not (tmp_path / "mesh_audit.jsonl.1").exists()
