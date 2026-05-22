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
    yield
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False


def _read_lines(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ─── Sequence numbers ────────────────────────────────────────────────────


class TestSequence:
    def test_sequence_starts_at_one_and_increments(self, tmp_path):
        audit.log_safety_event("a", "p1", {"x": 1})
        audit.log_safety_event("b", "p1", {"x": 2})
        records = audit.read_audit_log()
        assert [r["seq"] for r in records] == [1, 2]

    def test_sequence_per_peer(self):
        """seq is per-peer monotonic. Two peers writing concurrently each
        produce their own 1, 2, 3, ... sequence. The overall log can
        interleave them but per-peer adjacency is preserved — which is
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


# ─── HMAC signing ────────────────────────────────────────────────────────


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


# ─── Integrity verification ──────────────────────────────────────────────


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

    def test_mixed_signed_and_unsigned(self, monkeypatch):
        # First record: no PSK. Second: with PSK (rollout scenario).
        audit.log_safety_event("e", "p1", {})
        monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "topsecret")
        audit.log_safety_event("e", "p1", {})

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
        # Mutate the in-memory list — verify uses what we hand it.
        records.append({"ts": 0, "event": "extra", "peer_id": "p1", "payload": {}, "seq": 999})
        result = audit.verify_audit_integrity(records)
        assert result["total"] == 2
        assert result["sequence_gaps"] == [(1, 999)]
