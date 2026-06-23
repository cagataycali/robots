"""Forensic-walker defenses for :mod:`strands_robots.mesh.audit`.

``read_audit_log`` is both the operator-facing forensic reader and the
seed source for ``_load_seq_counters`` when the seq sidecar is corrupt.
That dual role makes its file-handling discipline security-relevant:

* A rotated log file swapped for a SYMLINK must be refused, not followed
  (mirrors the O_NOFOLLOW discipline applied to every other open in the
  module). Following it would let an attacker redirect the forensic read
  to ``/dev/null`` (fail-open seq reset) or to forged content.
* ``_audit_log_files_in_order`` must only treat files whose suffix after
  the active-log name is purely numeric as rotated copies; an unrelated
  sibling such as ``mesh_audit.jsonl.bak`` must be ignored so its bytes
  never enter the forensic stream.
* The ``since=`` cutoff must drop records strictly older than the
  timestamp (and records with a non-numeric ``ts``) while keeping newer
  ones, so a time-bounded forensic query returns only the window asked
  for.

These behaviors had no direct regression coverage; this file pins them.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolated_audit(monkeypatch, tmp_path):
    """Fresh audit dir + reset module state for each test."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    yield
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform without symlink support")
def test_read_audit_log_refuses_symlinked_rotated_file(tmp_path, monkeypatch, caplog):
    """A rotated log file that is a SYMLINK must be skipped, not followed.

    The active log is read normally; the symlinked rotated copy's target
    bytes must never appear in the returned records.
    """
    # Write a legitimate record to the active log.
    audit.log_safety_event("real_event", "peer-a", {"index": 1})
    active = audit.audit_log_path()
    assert active.exists()

    # Plant attacker-controlled bytes outside the audit dir, then point a
    # rotated-log path at them via a symlink.
    secret = tmp_path / "attacker_payload.jsonl"
    secret.write_text('{"event": "forged", "payload": {"index": 999}}\n', encoding="utf-8")
    rotated = active.with_suffix(active.suffix + ".1")
    os.symlink(secret, rotated)
    assert rotated.is_symlink()

    with caplog.at_level("WARNING"):
        records = audit.read_audit_log()

    events = [r.get("event") for r in records]
    assert "real_event" in events, "active log record should still be read"
    assert "forged" not in events, "symlinked rotated log must not be followed"
    assert any("refusing to read" in m and "SYMLINK" in m for m in caplog.messages), (
        f"expected a symlink-refusal warning, got {caplog.messages}"
    )


def test_files_in_order_ignores_non_numeric_suffix_siblings(tmp_path, monkeypatch):
    """Only ``<active>.<digits>`` siblings count as rotated copies.

    A sibling like ``mesh_audit.jsonl.bak`` must be excluded so its bytes
    never enter the forensic read stream.
    """
    audit.log_safety_event("real_event", "peer-a", {"index": 1})
    active = audit.audit_log_path()

    # A genuine rotated copy (numeric suffix) and a decoy (non-numeric).
    rotated = active.with_suffix(active.suffix + ".1")
    rotated.write_text('{"event": "rotated_event", "payload": {"index": 0}}\n', encoding="utf-8")
    decoy = active.with_suffix(active.suffix + ".bak")
    decoy.write_text('{"event": "decoy_event", "payload": {"index": -1}}\n', encoding="utf-8")

    ordered = audit._audit_log_files_in_order()
    names = [p.name for p in ordered]
    assert rotated.name in names, "numeric-suffix rotated copy must be included"
    assert decoy.name not in names, "non-numeric-suffix sibling must be excluded"
    # Chronological order: oldest rotated copy first, active log last.
    assert names[-1] == active.name

    events = [r.get("event") for r in audit.read_audit_log()]
    assert "rotated_event" in events
    assert "real_event" in events
    assert "decoy_event" not in events


def test_read_audit_log_since_filters_old_and_non_numeric_ts(tmp_path, monkeypatch):
    """``since=`` keeps records at/after the cutoff and drops older ones.

    A record whose ``ts`` is missing or non-numeric is also dropped, since
    it cannot be placed in the requested time window.
    """
    active = audit.audit_log_path()
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        "\n".join(
            [
                '{"event": "old", "ts": 100.0, "payload": {}}',
                '{"event": "new", "ts": 200.0, "payload": {}}',
                '{"event": "no_ts", "payload": {}}',
                '{"event": "bad_ts", "ts": "not-a-number", "payload": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = audit.read_audit_log(since=150.0)
    events = [r.get("event") for r in records]
    assert events == ["new"], f"since= should keep only records at/after the cutoff, got {events}"
