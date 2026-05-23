"""Audit-log HMAC integrity and sequence-gap detection tests.

These tests cover the on-disk forensic guarantees of
:mod:`strands_robots.mesh.audit`:

* Records gain a monotonic ``seq`` field.
* When STRANDS_MESH_AUDIT_PSK is set, every record carries a HMAC ``sig``.
* :func:`verify_audit_integrity` detects:
   - tampered payloads (sig mismatch)
   - sequence gaps (deleted records)
   - mixed signed/unsigned states (rollout in progress)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolated_audit(monkeypatch, tmp_path):
    """Each test gets a fresh audit dir and reset sequence counter."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False  # reset so tests are deterministic
    audit._AUDIT_STATE.psk_fingerprint = None  # reset PSK snapshot too
    yield
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_fingerprint = None


def _read_lines(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- Sequence numbers ----------------------------------------------------


class TestSequence:
    def test_sequence_starts_at_one_and_increments(self, tmp_path):
        audit.log_safety_event("a", "p1", {"x": 1})
        audit.log_safety_event("b", "p1", {"x": 2})
        records = audit.read_audit_log()
        assert [r["seq"] for r in records] == [1, 2]

    def test_sequence_per_peer(self):
        """seq is per-peer monotonic. Two peers writing concurrently each
        produce their own 1, 2, 3, ... sequence. The overall log can
        interleave them but per-peer adjacency is preserved -- which is
        what verify_audit_integrity's gap-detection relies on."""
        audit.log_safety_event("e", "peer-a", {})
        audit.log_safety_event("e", "peer-b", {})
        audit.log_safety_event("e", "peer-a", {})
        audit.log_safety_event("e", "peer-b", {})
        records = audit.read_audit_log()
        seq_by_peer: dict[str, list[int]] = {}
        for r in records:
            seq_by_peer.setdefault(r["peer_id"], []).append(r["seq"])
        assert seq_by_peer["peer-a"] == [1, 2]
        assert seq_by_peer["peer-b"] == [1, 2]
        # And verify no phantom gaps in the multi-peer case.
        assert audit.verify_audit_integrity()["sequence_gaps"] == []


# --- HMAC signing --------------------------------------------------------


class TestSigning:
    def test_no_psk_no_signature(self):
        audit.log_safety_event("e", "p1", {})
        records = audit.read_audit_log()
        assert "sig" not in records[0]

    def test_with_psk_signature_present(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {"x": 1})
        records = audit.read_audit_log()
        assert "sig" in records[0]
        assert len(records[0]["sig"]) == 64  # sha256 hex

    def test_signature_changes_with_payload(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {"x": 1})
        audit.log_safety_event("e", "p1", {"x": 2})
        records = audit.read_audit_log()
        assert records[0]["sig"] != records[1]["sig"]


# --- Integrity verification ----------------------------------------------


class TestVerifyIntegrity:
    def test_clean_log_verifies(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        for i in range(5):
            audit.log_safety_event("e", "p1", {"i": i})
        result = audit.verify_audit_integrity()
        assert result["ok"] is True
        assert result["total"] == 5
        assert result["signed"] == 5
        assert result["verified"] == 5
        assert result["bad_signature"] == 0
        assert result["sequence_gaps"] == []

    def test_tampered_payload_detected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {"x": 1})

        # Edit the file: change the payload but leave the original sig.
        p = audit.audit_log_path()
        records = _read_lines(p)
        records[0]["payload"] = {"x": 99}  # tamper
        p.write_text(json.dumps(records[0], separators=(",", ":")) + "\n")

        result = audit.verify_audit_integrity()
        assert result["ok"] is False
        assert result["bad_signature"] == 1

    def test_sequence_gap_detected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {"i": 1})
        audit.log_safety_event("e", "p1", {"i": 2})
        audit.log_safety_event("e", "p1", {"i": 3})

        # Delete the middle record.
        p = audit.audit_log_path()
        records = _read_lines(p)
        kept = [records[0], records[2]]
        p.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in kept) + "\n")

        result = audit.verify_audit_integrity()
        assert result["ok"] is False
        assert result["sequence_gaps"] == [(1, 3)]

    def test_mixed_signed_and_unsigned_via_separate_processes(self, monkeypatch, tmp_path):
        # Post-R18: a single process cannot transition unsigned -> signed
        # mid-run (raises AuditPSKDegradedError). The "mixed" log shape
        # only arises when separate processes write to the same audit
        # directory at different times. Simulate by crafting the file
        # contents directly so verify_audit_integrity is still exercised
        # against a mixed log shape.
        import json as _json

        log_path = audit.audit_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Unsigned record (process A, no PSK).
        rec1 = {"ts": 1.0, "event": "e", "peer_id": "p1", "payload": {}, "seq": 1}
        # Signed record (process B, with PSK). Compute the real signature.
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        rec2 = {"ts": 2.0, "event": "e", "peer_id": "p1", "payload": {}, "seq": 2}
        rec2["sig"] = audit._sign_record(rec2)
        log_path.write_text(_json.dumps(rec1) + "\n" + _json.dumps(rec2) + "\n")

        result = audit.verify_audit_integrity()
        assert result["psk_present"] is True
        assert result["missing_sig"] == 1
        assert result["signed"] == 1
        assert result["verified"] == 1

    def test_unverifiable_without_psk(self, monkeypatch):
        # Sign a record then verify with PSK gone (e.g. forensic reader has
        # access to the file but not the secret).
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {})
        monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")

        result = audit.verify_audit_integrity()
        assert result["psk_present"] is False
        assert result["signed"] == 1
        assert result["verified"] == 0
        assert result["bad_signature"] == 0

    def test_verify_gracefully_handles_empty_log(self):
        result = audit.verify_audit_integrity()
        assert result["total"] == 0
        assert result["ok"] is True

    def test_caller_can_supply_records(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {})
        records = audit.read_audit_log()
        # Mutate the in-memory list -- verify uses what we hand it.
        # Review feedback: an unsigned record under PSK is forged
        # by definition. The verifier MUST refuse to advance the
        # per-peer cursor past it, otherwise an attacker who omits the
        # sig field hides arbitrary deletions of subsequent records.
        # The gap (1 -> 999) therefore does NOT appear in sequence_gaps;
        # the forgery surfaces in missing_sig and ok=False instead.
        records.append({"ts": 0, "event": "extra", "peer_id": "p1", "payload": {}, "seq": 999})
        result = audit.verify_audit_integrity(records)
        assert result["total"] == 2
        assert result["missing_sig"] >= 1
        assert result["ok"] is False
        assert result["sequence_gaps"] == [], (
            f"unsigned record advanced the per-peer cursor and masked the forgery: {result}"
        )


# --- Rotation-spanning read coverage --------------------------------


def test_read_audit_log_spans_rotated_files(tmp_path, monkeypatch):
    """Reviewer finding: ``verify_audit_integrity`` was blind to records
    that lived in rotated logs (``mesh_audit.jsonl.1`` etc.). Force a
    rotation, then assert the reader returns events from BOTH the
    rotated copy and the active log.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "800")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "10")

    import importlib

    from strands_robots.mesh import audit

    importlib.reload(audit)

    # Write enough records to force at least one rotation. Each record
    # carries a unique index so we can verify ordering after the read.
    n = 30
    for i in range(n):
        audit.log_safety_event(
            "rotation_span_test",
            "peer-x",
            {"index": i, "filler": "x" * 32},
        )

    # Verify the rotation actually happened (at least one .jsonl.N file
    # alongside the active log).
    rotated = sorted(p.name for p in tmp_path.iterdir() if ".jsonl." in p.name)
    assert rotated, "test setup failed: no rotation produced"

    # Now read all records and confirm we see indices spanning the
    # rotation boundary -- specifically, the lowest indices live in
    # the rotated file and the highest in the active log.
    records = audit.read_audit_log()
    indices = [r.get("payload", {}).get("index") for r in records if r.get("event") == "rotation_span_test"]
    indices = [i for i in indices if isinstance(i, int)]

    assert indices, "no rotation_span_test records read"
    assert min(indices) < n // 2, (
        f"reader missed pre-rotation records: min index was {min(indices)}, expected to span 0..{n - 1}"
    )
    assert max(indices) >= n - 5, f"reader missed post-rotation records: max index was {max(indices)}"

    # And the records must come out in chronological order.
    assert indices == sorted(indices), "rotated + active records were not concatenated in chronological order"


def test_missing_sig_does_not_advance_cursor_when_psk_present(tmp_path, monkeypatch):
    """A forged record that simply
    omits the ``sig`` field used to silently advance the per-peer
    cursor. An attacker who could write to the audit log but did not
    have the PSK could therefore hide deletions of subsequent
    records by writing one unsigned record with an inflated ``seq``.

    Verify the fix: when the verifier has the PSK and an unsigned
    record is found, the per-peer cursor is NOT advanced and the
    forgery is reported in the ``missing_sig`` count.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "yin-replay-test")
    import importlib

    from strands_robots.mesh import audit

    importlib.reload(audit)

    # Two legitimate signed records.
    audit.log_safety_event("ok_event", "peer-x", {"i": 1})
    audit.log_safety_event("ok_event", "peer-x", {"i": 2})

    # Append an UNSIGNED record with an inflated seq value -- the
    # attack pattern called out in review. By writing seq=999 with no sig,
    # an attacker hopes the verifier will treat the next legitimate
    # record's seq as adjacent to the forged jump.
    log_path = audit.audit_log_path()
    forged = {
        "event": "forged",
        "peer_id": "peer-x",
        "seq": 999,  # fake jump
        "ts": 1.0,
        "payload": {"hidden": "deletion"},
        # NO "sig" field -- this is the attack
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(forged) + "\n")

    # Then a third legitimate record. With the bug, the cursor would
    # have jumped to 999 and the new record's seq=3 would look like
    # a regression rather than a gap.
    audit.log_safety_event("ok_event", "peer-x", {"i": 3})

    report = audit.verify_audit_integrity()

    # The forgery is counted.
    assert report["missing_sig"] >= 1, f"unsigned record was not flagged: {report}"
    # The verifier reports ok=False because PSK is present + missing_sig > 0.
    assert report["ok"] is False, f"verifier reported ok=True despite forged unsigned record: {report}"
    # The cursor did NOT jump to the forged seq=999, so the third
    # legitimate record (seq=3) is seen as adjacent to the second
    # legit record (seq=2) -- no gap reported.
    assert report["sequence_gaps"] == [], f"cursor advanced past forged record, masking real gap: {report}"


def test_log_safety_event_does_not_propagate_seq_errors(monkeypatch, tmp_path, caplog):
    """Review feedback: log_safety_event's docstring says 'Raises: Nothing'
    but _next_seq used to be invoked outside the try/except. Verify the fix:
    even when _next_seq raises, log_safety_event swallows and logs a WARNING.
    """
    import logging

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)

    # Force _next_seq to raise.
    def boom(peer_id):
        raise OSError("simulated audit-dir failure")

    monkeypatch.setattr(audit, "_next_seq", boom)

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
        # Must NOT raise.
        audit.log_safety_event("test_event", "peer-x", {"k": "v"})

    # Warning was emitted.
    assert any("_next_seq failed" in record.message for record in caplog.records), (
        f"expected _next_seq failure WARNING in {[r.message for r in caplog.records]}"
    )

    # No record was written.
    log_path = audit.audit_log_path()
    if log_path.exists():
        assert log_path.read_text().strip() == "", "record was written despite _next_seq failure"


def test_psk_degrade_drops_record(monkeypatch, tmp_path, caplog):
    """Pin regression: AuditPSKDegradedError blocks silent unsigned downgrade.

    Scenario: PSK is set at first write (audit log starts signed), then
    PSK is unset mid-run. The second write MUST be dropped (not written
    unsigned) and an ERROR must be logged. This pins the AuditPSKDegradedError
    code path in _sign_record.

    Without this test, a refactor that moves the PSK snapshot logic or
    accidentally inverts the comparison silently reintroduces the
    unsigned-downgrade path and the test suite stays green.
    """
    import logging

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    # Reset audit state for this test.
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._SEQ_COUNTERS.clear()

    # Phase 1: PSK is set, first record is signed.
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "test-psk-secret")
    audit.log_safety_event("signed_event", "peer-a", {"phase": "one"})

    # Verify first record was written and is signed.
    records = list(audit.read_audit_log())
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    assert records[0].get("sig") is not None, "first record must be signed"
    assert records[0]["event"] == "signed_event"

    # Phase 2: PSK is unset mid-run (simulates attacker or accident).
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK")

    with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.audit"):
        # Must NOT raise (audit failures must not crash the safety path).
        # R19: writes a poison record with sig=PSK_DEGRADED instead of
        # silent drop, preserving the forensic trail.
        audit.log_safety_event("unsigned_attempt", "peer-a", {"phase": "two"})

    # R19: poison record IS written (was previously dropped). The poison
    # record has sig="PSK_DEGRADED" which verify_audit_integrity reports
    # as bad_signature, surfacing the transition to forensics.
    records_after = list(audit.read_audit_log())
    assert len(records_after) == 2, f"expected 2 records (signed + poison), got {len(records_after)}"
    poison = [r for r in records_after if r.get("event") == "unsigned_attempt"]
    assert len(poison) == 1
    assert poison[0].get("sig") == "PSK_DEGRADED"
    assert "psk_degraded" in poison[0]

    # Error must have been logged mentioning PSK degradation.
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("PSK" in msg or "AuditPSKDegradedError" in msg or "degrade" in msg.lower() for msg in error_messages), (
        f"expected PSK degradation ERROR in logs, got: {error_messages}"
    )


def test_psk_degrade_unsigned_to_signed_drops_record(monkeypatch, tmp_path, caplog):
    """Pin regression: PSK upward-transition is also refused (B3 from R18 audit).

    Scenario: process boots with PSK unset (writes unsigned records),
    then the PSK is installed mid-run. The next record MUST be dropped --
    a verifier cannot distinguish "PSK rolled out mid-run" from
    "attacker briefly cleared the PSK to forge unsigned records, then
    restored it to evade detection". Same forensic posture as the
    signed-then-unsigned direction (test_psk_degrade_drops_record).

    Without this pin a future refactor that drops the symmetric branch
    silently re-opens the forgery window between PSK rotations.
    """
    import logging

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    # Reset audit state.
    audit._AUDIT_STATE.psk_fingerprint = None
    audit._AUDIT_STATE.seq_loaded = False
    audit._SEQ_COUNTERS.clear()

    # Phase 1: PSK is unset, first record is unsigned.
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit.log_safety_event("unsigned_event", "peer-a", {"phase": "one"})

    records = list(audit.read_audit_log())
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    assert records[0].get("sig") is None, "first record must be unsigned"
    assert records[0]["event"] == "unsigned_event"

    # Phase 2: PSK installed mid-run (rollout OR attacker restoring it).
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "test-psk-secret")

    with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.audit"):
        audit.log_safety_event("signed_attempt", "peer-a", {"phase": "two"})

    # R19: poison record IS written (was previously dropped). Symmetric
    # with the signed->unsigned direction.
    records_after = list(audit.read_audit_log())
    assert len(records_after) == 2, (
        f"expected 2 records (unsigned + poison) on PSK install mid-run, got {len(records_after)}"
    )
    poison = [r for r in records_after if r.get("event") == "signed_attempt"]
    assert len(poison) == 1
    assert poison[0].get("sig") == "PSK_DEGRADED"
    assert "psk_degraded" in poison[0]

    # Error must have been logged mentioning PSK degradation.
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("PSK" in msg or "AuditPSKDegradedError" in msg or "degrade" in msg.lower() for msg in error_messages), (
        f"expected PSK degradation ERROR in logs, got: {error_messages}"
    )


def test_cursor_does_not_roll_backward_on_forged_low_seq(monkeypatch, tmp_path):
    """Pin regression: per-peer cursor refuses backward rolls.

    Threat model (no PSK case): an attacker who can write to a rotated
    .jsonl.N file inserts a record with a seq value LOWER than the
    current cursor. Without a guard, ``last_seq_by_peer[peer] = seq``
    rolls the cursor backward; the next legit record then looks adjacent
    to the forged low-seq record (no gap reported) and any real records
    deleted between the forged seq and the legit record are silently
    masked.

    Pin: after a forged record with seq < prev, the cursor must NOT roll
    back. The next legit record is then evaluated against the highest
    seq seen, surfacing the gap.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    import importlib

    from strands_robots.mesh import audit

    importlib.reload(audit)

    log_path = audit.audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Hand-craft an unsigned log (no PSK). Three records from peer-x:
    # seq=1 (legit), seq=2 (legit), then a forged record claiming seq=1
    # again (cursor rollback attempt), then a legit seq=3.
    records = [
        {"event": "e1", "peer_id": "peer-x", "seq": 1, "ts": 1.0, "payload": {}},
        {"event": "e2", "peer_id": "peer-x", "seq": 2, "ts": 2.0, "payload": {}},
        # FORGED: seq=1 again (rollback attempt). Real records seq=3..7
        # would have been here; attacker deleted them.
        {"event": "forged", "peer_id": "peer-x", "seq": 1, "ts": 3.0, "payload": {}},
        # Legit seq=8 -- if cursor rolled back to 1, this looks like a
        # gap of (1, 8); if the cursor held at 2 (correct behavior),
        # this is also reported as a gap (2, 8) -- either way a gap is
        # surfaced. The bug is that a backward-rolled cursor would
        # report the smaller, misleading gap.
        {"event": "e8", "peer_id": "peer-x", "seq": 8, "ts": 4.0, "payload": {}},
    ]
    with open(log_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    report = audit.verify_audit_integrity()

    # The cursor must NOT have rolled back to 1. The final gap must be
    # reported against the highest legit seq seen (2), not the forged
    # seq (1). With the bug, the gap would be (1, 8); with the fix it
    # is (2, 8) AND (2, 1) (the forged-low jump itself).
    gaps = report["sequence_gaps"]
    assert gaps, f"no gaps reported despite forged low-seq + missing records: {report}"
    # The legit-record gap (2 -> 8) must be present with the correct prev.
    legit_gap = next((g for g in gaps if g[1] == 8), None)
    assert legit_gap is not None, f"gap to seq=8 not surfaced: {gaps}"
    assert legit_gap[0] == 2, (
        f"cursor rolled back to forged seq -- gap reports prev={legit_gap[0]} "
        f"instead of expected prev=2 (highest legit seq before the forgery)"
    )
