"""Pin: ``_persist_seq_counters`` stays fail-soft when durability syscalls fail.

The audit sequence-counter sidecar is written with a defence-in-depth
durability sequence: ``fsync`` on the data fd, ``fsync`` on the parent
directory after the atomic ``os.replace``, and a ``chmod`` to ``0o600``.
On real deployment surfaces that reject these syscalls -- FAT32 and many
mounted volumes reject ``fsync``, NFS-without-uid-map and restricted mount
options reject ``chmod`` -- audit persistence is fail-soft *by contract*:
it must still produce a valid, round-tripping sidecar rather than crash the
safety code path.

The symmetric symlink/atomic-write behaviour is pinned in
``test_audit_seq_persist_symlink.py``; these tests pin the durability
fail-soft branches specifically. Pre-fix (i.e. if any of the
``try: ... except OSError: pass`` guards around ``fsync``/``chmod`` were
removed), each of these would propagate the syscall's ``OSError`` out of
``_persist_seq_counters`` and the sidecar would be missing or unreadable.
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


def _read_sidecar() -> dict:
    sidecar = audit._seq_sidecar_path()
    return json.loads(sidecar.read_text(encoding="utf-8"))


def test_persist_survives_data_fsync_rejection(monkeypatch) -> None:
    """A filesystem that rejects ``fsync`` on the data fd still gets a sidecar.

    FAT32 and several mounted volumes raise ``OSError`` from ``fsync``. The
    write must complete (data is in the page cache) and round-trip; durability
    is weaker than ideal but the safety path stays alive.
    """
    real_fsync = os.fsync

    def _fsync_rejects_files(fd):
        # Reject fsync on the data file fd (a regular file), but allow the
        # parent-directory fsync (a directory fd) through so this test
        # isolates the data-fd branch.
        if os.fstat(fd).st_mode & 0o170000 == 0o100000:  # S_IFREG
            raise OSError("fsync not supported on this filesystem")
        return real_fsync(fd)

    monkeypatch.setattr(audit.os, "fsync", _fsync_rejects_files)

    audit._SEQ_COUNTERS["peerA"] = 7
    audit._persist_seq_counters()  # must not raise

    assert _read_sidecar() == {"peerA": 7}


def test_persist_survives_dir_fsync_rejection(monkeypatch) -> None:
    """A filesystem that rejects directory ``fsync`` still gets a sidecar."""
    real_fsync = os.fsync

    def _fsync_rejects_dirs(fd):
        if os.fstat(fd).st_mode & 0o170000 == 0o040000:  # S_IFDIR
            raise OSError("directory fsync not supported")
        return real_fsync(fd)

    monkeypatch.setattr(audit.os, "fsync", _fsync_rejects_dirs)

    audit._SEQ_COUNTERS["peerB"] = 13
    audit._persist_seq_counters()  # must not raise

    assert _read_sidecar() == {"peerB": 13}
    # The atomic tmp file must have been renamed away, not left behind.
    sidecar = audit._seq_sidecar_path()
    assert not sidecar.with_suffix(sidecar.suffix + ".tmp").exists()


def test_persist_survives_chmod_rejection(monkeypatch, caplog) -> None:
    """A filesystem that rejects ``chmod`` still produces a readable sidecar.

    FAT32 / NFS-without-uid-map / restricted mounts reject ``chmod``. The
    sidecar is still written and readable; we'd rather lose the 0o600 mode
    than crash safety persistence. The inner ``except OSError`` around
    ``chmod`` also keeps the outer fail-soft handler from logging a
    misleading "could not persist" warning for a sidecar that *was* written.
    """

    def _chmod_boom(*_args, **_kwargs):
        raise OSError("chmod not permitted on this filesystem")

    monkeypatch.setattr(audit.os, "chmod", _chmod_boom)

    audit._SEQ_COUNTERS["peerC"] = 21
    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._persist_seq_counters()  # must not raise

    sidecar = audit._seq_sidecar_path()
    assert sidecar.exists()
    assert _read_sidecar() == {"peerC": 21}
    # A successfully-written sidecar must not be mislabelled as a failure.
    assert not any("could not persist seq sidecar" in rec.message for rec in caplog.records), (
        "chmod rejection must not surface as a persistence failure"
    )


def test_persist_survives_dir_open_rejection(monkeypatch) -> None:
    """An unreadable parent dir (open fails) still leaves the sidecar in place.

    The parent-directory fsync is best-effort: if ``os.open`` on the dir
    raises, ``dir_fd`` is ``None`` and the durability step is skipped without
    disturbing the already-renamed sidecar.
    """
    real_open = os.open
    sidecar_parent = str(audit._seq_sidecar_path().parent)

    def _open_rejects_parent_dir(path, flags, *args, **kwargs):
        if str(path) == sidecar_parent and (flags & os.O_RDONLY == os.O_RDONLY) and not (flags & os.O_WRONLY):
            raise OSError("permission denied opening dir for fsync")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit.os, "open", _open_rejects_parent_dir)

    audit._SEQ_COUNTERS["peerD"] = 4
    audit._persist_seq_counters()  # must not raise

    assert _read_sidecar() == {"peerD": 4}


def test_persist_durability_failures_still_reload(monkeypatch) -> None:
    """End-to-end: counters persisted under syscall rejection reload cleanly."""

    def _chmod_boom(*_args, **_kwargs):
        raise OSError("chmod not permitted")

    monkeypatch.setattr(audit.os, "chmod", _chmod_boom)

    audit._SEQ_COUNTERS["peerX"] = 30
    audit._SEQ_COUNTERS["peerY"] = 31
    audit._persist_seq_counters()

    # Simulate a fresh process: drop the cache and reload from disk.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._load_seq_counters()

    assert audit._SEQ_COUNTERS.get("peerX") == 30
    assert audit._SEQ_COUNTERS.get("peerY") == 31


def test_persist_fails_soft_when_data_write_raises(monkeypatch, caplog) -> None:
    """A write error mid-``json.dump`` (e.g. ENOSPC) fails soft, leaves no sidecar.

    The sidecar is written to a temp file under a ``with os.fdopen(...)`` block
    and only atomically renamed into place once the bytes are down. If the
    write itself raises (disk full, quota exceeded), the inner handler must
    close the orphaned descriptor and re-raise so the outer fail-soft guard
    logs and returns -- WITHOUT renaming a half-formed temp into the real
    sidecar. A leftover empty/partial sidecar would otherwise be loaded as
    authoritative on the next boot and roll every peer's seq counter back to
    zero, defeating replay-adjacency in ``verify_audit_integrity``.

    Pre-fix (i.e. if the ``raise`` after the descriptor-close were dropped),
    control would fall through to ``os.replace`` and rename the empty temp
    over the sidecar, so ``sidecar.exists()`` would flip True and this fails.
    """

    def _dump_boom(*_args, **_kwargs):
        raise OSError("ENOSPC: no space left on device")

    monkeypatch.setattr(audit.json, "dump", _dump_boom)

    audit._SEQ_COUNTERS["peerE"] = 9
    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._persist_seq_counters()  # must not raise

    sidecar = audit._seq_sidecar_path()
    assert not sidecar.exists(), "a failed write must not leave a sidecar that could roll counters back"
    assert any("could not persist seq sidecar" in rec.getMessage() for rec in caplog.records), (
        "a write failure must surface via the fail-soft persistence warning"
    )


def test_persist_closes_orphaned_fd_when_fdopen_raises(monkeypatch, caplog) -> None:
    """``os.fdopen`` raising before adopting the fd must not leak the descriptor.

    On the rare path where ``os.fdopen`` raises (e.g. EMFILE while building the
    buffered wrapper) the raw fd from ``os.open`` was never handed to a file
    object, so the context manager cannot close it. The defence-in-depth
    handler must close that orphaned fd explicitly and re-raise into the outer
    fail-soft guard. Pre-fix (no explicit ``os.close``), the descriptor would
    leak on every persist failure under fd pressure.
    """
    real_close = audit.os.close
    closed_fds: list[int] = []

    def _fdopen_boom(*_args, **_kwargs):
        raise OSError("EMFILE: too many open files")

    def _tracking_close(fd: int) -> None:
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(audit.os, "fdopen", _fdopen_boom)
    monkeypatch.setattr(audit.os, "close", _tracking_close)

    audit._SEQ_COUNTERS["peerF"] = 11
    with caplog.at_level("WARNING", logger="strands_robots.mesh.audit"):
        audit._persist_seq_counters()  # must not raise

    sidecar = audit._seq_sidecar_path()
    assert not sidecar.exists(), "a failed fdopen must not leave a sidecar in place"
    assert closed_fds, "the orphaned fd from a failed fdopen must be closed (no descriptor leak)"
    assert any("could not persist seq sidecar" in rec.getMessage() for rec in caplog.records), (
        "an fdopen failure must surface via the fail-soft persistence warning"
    )
