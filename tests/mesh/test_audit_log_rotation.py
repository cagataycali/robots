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
