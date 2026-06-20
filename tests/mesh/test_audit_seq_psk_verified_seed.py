"""Pin: the audit-log seq-seed fallback only trusts HMAC-verified records.

When the seq sidecar is unusable (missing/corrupt/symlinked) the audit module
re-seeds ``_SEQ_COUNTERS`` by walking the persisted audit log and taking
``max(seq)`` per peer. That walk is a trust boundary: it restores the
monotonic-sequence floor that replay protection depends on.

The circular-trust defence (``_load_seq_counters`` in
``strands_robots.mesh.audit``) closes a self-poisoning hole on that boundary:
when ``STRANDS_MESH_AUDIT_PSK`` is configured, the seed walk trusts ONLY records
whose HMAC ``sig`` it can recompute and ``compare_digest``-match. Otherwise an
attacker who could append a single forged record (no PSK in dev, or a PSK
exfiltrated long enough to write one line then cleared) could plant
``seq=10**9`` and, on the next restart with a degraded sidecar, that forged
value would silently become the per-peer floor -- denying the legitimate writer
~a billion working sequence numbers.

These tests exercise the PSK-present branch specifically (the existing recovery
test covers only the no-PSK posture):

* a correctly-signed record seeds the counter,
* an unsigned record and a ``PSK_DEGRADED`` poison sentinel are skipped,
* a wrong-HMAC record is skipped,
* an over-cap but correctly-signed record is refused by the shared cap,
* a log of only-unverified records seeds nothing (fail-closed, not fail-open).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from strands_robots.mesh import audit


@pytest.fixture
def isolated_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the audit module at an empty per-test dir and reset its state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    yield tmp_path
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False


def _sign(record: dict, psk: bytes) -> str:
    """Recompute the record's canonical HMAC the way the module does."""
    return hmac.new(psk, audit._canonical_bytes(record), hashlib.sha256).hexdigest()


def _write_log(records: list[dict]) -> None:
    """Write records to the active audit log; NO sidecar (degraded entry)."""
    log_path = audit.audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_psk_verified_record_seeds_counter(isolated_audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A correctly-signed audit record seeds the seq floor under a PSK."""
    psk = b"seed-key-very-secret"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk.decode())

    rec = {"ts": 1000.0, "peer_id": "peer-A", "seq": 7, "event": "estop"}
    rec["sig"] = _sign(rec, psk)
    _write_log([rec])

    audit._load_seq_counters()

    assert audit._SEQ_COUNTERS.get("peer-A") == 7
    # _next_seq advances past the seeded floor rather than fail-open to 1.
    assert audit._next_seq("peer-A") == 8


def test_unsigned_and_poison_records_are_not_trusted(isolated_audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a PSK, records lacking a valid HMAC sig do NOT seed the floor."""
    psk = b"seed-key-very-secret"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk.decode())

    unsigned = {"ts": 1.0, "peer_id": "peer-A", "seq": 500, "event": "forged"}
    poison = {"ts": 2.0, "peer_id": "peer-A", "seq": 600, "event": "x", "sig": "PSK_DEGRADED"}
    signed = {"ts": 3.0, "peer_id": "peer-A", "seq": 9, "event": "real"}
    signed["sig"] = _sign(signed, psk)
    _write_log([unsigned, poison, signed])

    audit._load_seq_counters()

    # Only the verified seq=9 seeds; the unsigned 500 / poison 600 are ignored.
    assert audit._SEQ_COUNTERS.get("peer-A") == 9


def test_wrong_hmac_record_is_rejected(isolated_audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record signed with the wrong key fails compare_digest and is skipped."""
    real_psk = b"the-real-key"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", real_psk.decode())

    forged = {"ts": 1.0, "peer_id": "peer-B", "seq": 4242, "event": "forged"}
    forged["sig"] = _sign(forged, b"attacker-guessed-key")
    _write_log([forged])

    audit._load_seq_counters()

    assert "peer-B" not in audit._SEQ_COUNTERS
    assert audit._next_seq("peer-B") == 1  # fail-closed: no poisoned floor


def test_over_cap_signed_record_refused(
    isolated_audit: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Even a correctly-signed record above the cap is refused (shared cap)."""
    psk = b"seed-key-very-secret"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk.decode())

    over = {"ts": 1.0, "peer_id": "peer-C", "seq": audit._MAX_SEED_SEQ + 1, "event": "x"}
    over["sig"] = _sign(over, psk)
    _write_log([over])

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
        audit._load_seq_counters()

    assert "peer-C" not in audit._SEQ_COUNTERS
    assert any("peer-C" in r.getMessage() and "forged" in r.getMessage() for r in caplog.records)


def test_all_unverified_log_seeds_nothing(
    isolated_audit: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A log of only-unverified records seeds zero counters (fail-closed)."""
    psk = b"seed-key-very-secret"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", psk.decode())

    forgeries = [
        {"ts": 1.0, "peer_id": "peer-A", "seq": 100, "event": "forged1"},
        {"ts": 2.0, "peer_id": "peer-B", "seq": 200, "event": "forged2", "sig": "SIGN_FAILED"},
    ]
    _write_log(forgeries)

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
        audit._load_seq_counters()

    assert audit._SEQ_COUNTERS == {}
    assert any("unverified" in r.getMessage() and "NOT seeded" in r.getMessage() for r in caplog.records)
