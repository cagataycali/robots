"""Pin: the seq-counter restore and lock path fail soft, never crash.

``_next_seq`` sits on the mesh replay-protection hot path: every safety event
draws a monotonic sequence number, and restoring the per-peer floor after a
process restart is what stops an attacker from replaying an old signed command.
That restore (``_load_seq_counters``) and the inter-process lock that guards it
(``_seq_flock``) run in environments the writer does not control -- a degraded
audit directory, a filesystem that errors mid-read, or a platform without
``fcntl``. In all of those the contract is the same: degrade gracefully (log and
carry on) rather than raise out of the safety path.

These tests pin that fail-soft contract:

* ``_seq_flock`` on a host without ``fcntl`` yields an in-process-only lock
  instead of touching the unavailable syscall,
* ``_load_seq_counters`` is idempotent -- a second call after the one-shot load
  flag is set returns without re-reading the sidecar,
* when the sidecar is degraded AND the audit-log seed fallback read itself
  raises, ``_load_seq_counters`` logs and returns instead of propagating, and
  ``_next_seq`` keeps handing out sequence numbers.
"""

from __future__ import annotations

import logging
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


def test_seq_flock_without_fcntl_yields_in_process_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a host lacking ``fcntl`` the lock is a no-op that still runs the block.

    POSIX gets a real inter-process flock; Windows / restricted platforms have
    no ``fcntl`` and must fall back to the intra-process lock only, not crash on
    the missing syscall.
    """
    monkeypatch.setattr(audit, "_HAS_FCNTL", False)

    ran = False
    with audit._seq_flock():
        ran = True
    assert ran, "the flock context body must execute even with no fcntl backend"


def test_load_seq_counters_is_idempotent(isolated_audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second load after the one-shot flag is set returns without re-reading.

    The load walk is O(records); repeating it on every safety event would burn
    CPU on the hot path, so ``seq_loaded`` gates it to once per process.
    """
    audit._AUDIT_STATE.seq_loaded = True

    def _boom() -> Path:
        raise AssertionError("_seq_sidecar_path must not be touched on the idempotent early return")

    monkeypatch.setattr(audit, "_seq_sidecar_path", _boom)

    # Must not raise: the early return fires before the sidecar path is resolved.
    audit._load_seq_counters()


def test_seed_failsoft_when_audit_log_read_raises(
    isolated_audit: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A degraded sidecar plus an erroring log read logs and carries on.

    With no sidecar the restore falls through to the audit-log seed walk; if that
    read itself raises (disk fault, truncated log), the module must log a warning,
    mark the walk done, and leave the counters untouched -- not let the OSError
    escape onto the safety path.
    """

    def _raise(*_a: object, **_k: object) -> list[dict]:
        raise OSError("simulated audit-log read failure")

    monkeypatch.setattr(audit, "read_audit_log", _raise)

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.audit"):
        audit._load_seq_counters()  # must not raise

    assert audit._SEQ_COUNTERS == {}
    assert audit._AUDIT_STATE.audit_log_seeded is True
    assert any("could not seed from audit log" in r.getMessage() for r in caplog.records)


def test_next_seq_survives_degraded_audit_dir(isolated_audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_next_seq`` keeps issuing numbers when the audit-log seed read errors.

    This is the end-to-end safety-path guarantee: a broken audit directory must
    not take down sequence-number issuance (and thus every safety event).
    """

    def _raise(*_a: object, **_k: object) -> list[dict]:
        raise OSError("simulated audit-log read failure")

    monkeypatch.setattr(audit, "read_audit_log", _raise)

    assert audit._next_seq("peer-X") == 1
    assert audit._next_seq("peer-X") == 2
