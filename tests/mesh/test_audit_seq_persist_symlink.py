"""Pin: ``_persist_seq_counters`` writes atomically and refuses a symlink.

``_persist_seq_counters`` is the security-critical writer half of the audit
sequence-counter sidecar. Its symmetric reader half ``_load_seq_counters``
already has symlink-refusal coverage (see
``test_audit_seq_symlink.py``), but the writer side -- the atomic
``tmp + os.replace`` write, the 0o600 mode, and the ``is_symlink()``
refusal that defends against a swap-and-write redirect -- was previously
exercised only indirectly through ``_next_seq``. These tests pin the
contract directly:

* a normal write produces a real (non-symlink) sidecar with private
  permissions whose JSON round-trips back into ``_SEQ_COUNTERS``;
* if the sidecar path is a pre-planted symlink, the write fails soft
  (warns, does not follow the link, leaves the attacker target
  untouched) -- the same threat model as the reader-side defence;
* a write that hits an OSError degrades gracefully (warns, never raises)
  because audit persistence is fail-soft by contract.
"""

from __future__ import annotations

import json
import os

import pytest

from strands_robots.mesh import audit


@pytest.fixture(autouse=True)
def _isolate_audit_state(tmp_path, monkeypatch):
    """Each test starts with a fresh audit dir + reset module state."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()
    yield
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._SEQ_COUNTERS.clear()


def test_persist_writes_real_private_sidecar(tmp_path) -> None:
    """A normal persist writes a real file, 0o600, that round-trips."""
    audit._SEQ_COUNTERS["peerA"] = 7
    audit._SEQ_COUNTERS["peerB"] = 42

    audit._persist_seq_counters()

    sidecar = audit._seq_sidecar_path()
    assert sidecar.exists()
    assert not sidecar.is_symlink()
    # No leftover temp file from the atomic tmp + os.replace write.
    assert not sidecar.with_suffix(sidecar.suffix + ".tmp").exists()
    # Private permissions on POSIX (chmod is best-effort elsewhere).
    if os.name == "posix":
        assert (sidecar.stat().st_mode & 0o777) == 0o600
    # Disk contents round-trip back to the in-memory counters.
    restored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert restored == {"peerA": 7, "peerB": 42}


def test_persist_creates_missing_parent_dir(tmp_path, monkeypatch) -> None:
    """Persist creates the audit dir if it does not yet exist."""
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(nested))
    audit._SEQ_COUNTERS["peerC"] = 3

    audit._persist_seq_counters()

    sidecar = audit._seq_sidecar_path()
    assert sidecar.parent == nested
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"peerC": 3}


@pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt",
    reason="symlink semantics differ on Windows; O_NOFOLLOW is 0 there",
)
def test_persist_refuses_symlinked_sidecar(tmp_path, caplog) -> None:
    """A symlink at the sidecar path must not be followed when writing.

    Pre-fix, the writer would happily open the symlink and clobber the
    attacker-chosen target (or null-route the counters through e.g.
    ``/dev/null``). The ``is_symlink()`` check fails soft instead.
    """
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    attacker_target = attacker_dir / "evil.json"
    attacker_target.write_text(json.dumps({"untouched": True}), encoding="utf-8")

    sidecar = audit._seq_sidecar_path()
    os.symlink(attacker_target, sidecar)

    audit._SEQ_COUNTERS["peerA"] = 99
    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._persist_seq_counters()

    # The attacker's target file must be left byte-for-byte intact.
    assert json.loads(attacker_target.read_text(encoding="utf-8")) == {"untouched": True}
    # A WARNING must surface so operators can attribute the no-op.
    assert any("SYMLINK" in rec.message or "symlink" in rec.message.lower() for rec in caplog.records), (
        "expected a WARNING line about the symlinked sidecar"
    )


def test_persist_fails_soft_on_oserror(tmp_path, monkeypatch, caplog) -> None:
    """An OSError during the write degrades to a warning, never raises."""
    audit._SEQ_COUNTERS["peerA"] = 5

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    # Force the first filesystem touch inside the try-block to fail.
    monkeypatch.setattr(audit.os, "open", _boom)

    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._persist_seq_counters()  # must not raise

    assert any("could not persist seq sidecar" in rec.message for rec in caplog.records), (
        "expected a fail-soft WARNING when the sidecar write hits an OSError"
    )


def test_persist_then_load_round_trips(tmp_path) -> None:
    """End-to-end: persisted counters reload into a fresh in-memory state."""
    audit._SEQ_COUNTERS["peerX"] = 11
    audit._SEQ_COUNTERS["peerY"] = 22
    audit._persist_seq_counters()

    # Simulate a fresh process: drop the cache and the loaded flag.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._load_seq_counters()

    assert audit._SEQ_COUNTERS.get("peerX") == 11
    assert audit._SEQ_COUNTERS.get("peerY") == 22
