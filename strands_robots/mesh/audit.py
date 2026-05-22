"""Append-only audit log for safety-critical mesh events.

Safety actions on a multi-robot mesh (most importantly :func:`emergency_stop`)
need a tamper-evident trail that lives independently of stdout, structured
loggers, or any process that may crash mid-event.  This module owns that
trail.

Layout
------
By default the log lives at ``~/.strands_robots/mesh_audit.jsonl`` with
file mode ``0o600`` (owner read/write only) and the parent directory at
``0o700``.  The location can be overridden with the
``STRANDS_MESH_AUDIT_DIR`` environment variable; the JSONL file is always
named ``mesh_audit.jsonl``.

Format
------
Each line is one JSON object with these keys:

* ``ts`` — UNIX timestamp (float seconds, UTC)
* ``event`` — short event type, e.g. ``"emergency_stop"``
* ``peer_id`` — the mesh peer that owned the event
* ``payload`` — free-form dict with event-specific fields
* ``seq`` — process-monotonic sequence number. Useful for detecting
  truncation: gaps within a single peer's stream indicate missing events.
* ``sig`` — HMAC-SHA256 hex over the rest of the record. Present only when
  ``STRANDS_MESH_AUDIT_PSK`` is configured. Verifies that the record
  content has not been edited after write.

Integrity verification
----------------------
The audit log is the forensic trail for emergency stops, command
rejections, and resume attempts. To frustrate post-incident tampering by a
compromised process the writer attaches a per-record HMAC signature when
``STRANDS_MESH_AUDIT_PSK`` is set. :func:`verify_audit_integrity` walks the
log and reports:

* records with broken signatures (content was edited or partially
  truncated mid-line),
* sequence gaps (records were deleted),
* records lacking a signature (mixed-mode log, expected during rollouts).

The PSK lives in env / Secrets Manager, never in the file. A reader that
does not have the PSK can still read events; it just cannot verify them.

The file is opened in append mode for every write so concurrent writers
from multiple threads or processes never overwrite each other; ordering
across processes is best-effort.

Reading
-------
:func:`read_audit_log` parses the file line by line and returns a list of
event dicts.  Lines that fail to parse are silently skipped (defensive: the
audit log is forward-compatible with future fields).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# fcntl is POSIX-only. Windows deployments fall back to in-process
# locking and lose the cross-process safety guarantee documented in
# the module docstring.
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

_LOG_FILE_NAME = "mesh_audit.jsonl"
_DEFAULT_DIR = Path.home() / ".strands_robots"

# Audit-log rotation (Phase-4 / Cycle 6 / E1).
#
# Without a size cap, an attacker who can publish to a peer's
# safety/event topic (or, in permissive mode, anyone on the LAN) can
# fill the robot's disk by spamming events at line-rate. We cap the
# active log at ``_DEFAULT_LOG_MAX_BYTES`` and rotate to a numbered
# suffix on overflow. ``_DEFAULT_LOG_MAX_FILES`` rotated copies are
# kept; the oldest is discarded. Operators tune via
# ``STRANDS_MESH_AUDIT_MAX_BYTES`` and ``STRANDS_MESH_AUDIT_MAX_FILES``.
#
# We deliberately don't use logging.handlers.RotatingFileHandler — that
# class doesn't honour O_NOFOLLOW and would re-open the file via the
# default open() on rollover, defeating the F2 symlink guard. The
# rotation here uses os.rename + os.open(O_NOFOLLOW) consistently.
_DEFAULT_LOG_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_LOG_MAX_FILES: int = 5
_LOG_MAX_BYTES_CAP: int = 10 * 1024 * 1024 * 1024  # 10 GiB hard upper bound
_LOG_MAX_FILES_CAP: int = 100

# Serialise writes inside a single process so two threads can't interleave
# bytes inside one append. Different processes still need filesystem-level
# atomicity (one open(..., "a") write per event).
_WRITE_LOCK = threading.Lock()

# Per-peer monotonic sequence counters. Each peer_id has its own counter so
# the (peer_id, seq) pair is unique within one process AND consecutive
# values within a single peer's stream are guaranteed to be adjacent. This
# makes :func:`verify_audit_integrity` gap detection meaningful even in
# processes that host multiple Mesh peers (test harnesses, ``Simulation``).
#
# Counters persist to a sidecar file (``mesh_audit.seq.json`` next to the
# audit log) so a process restart does NOT reset them. Without the
# sidecar, a compromised process could delete records, restart, and
# yield a clean ``verify_audit_integrity()`` because every peer's seq
# would start over at 1 — defeating the gap-detection half of the
# threat model. The sidecar is reloaded inside the cross-process
# lockfile (``mesh_audit.seq.lock``, R5-3) on every ``_next_seq`` call
# and rewritten before the lock is released, so two processes sharing
# the same ``STRANDS_MESH_AUDIT_DIR`` cannot roll the counter back.
# Writes are fail-soft: a write failure logs at WARNING but does not
# break the safety code path.
#
# Cross-process guarantee: POSIX (``fcntl.flock``) only. Windows
# deployments fall back to in-process locking; running multiple
# writer processes against the same audit dir on Windows is not
# safe and not supported.
#
# R4-7 — audit-write amplification (accepted limitation): every event
# triggers a synchronous double write (audit line + sidecar
# tmp+os.replace+chmod+fsync). The token bucket caps inbound floods at
# ``STRANDS_MESH_PEER_RATE`` (default 20/60s/sender, max 1000 burst), so
# the worst-case write rate is bounded. If your deployment runs
# pathologically high audit volumes, batch the sidecar persistence by
# subclassing ``_persist_seq_counters`` to write at most once per N
# events with an atexit flush — the on-disk counter can then lose at
# most that many seconds of seq state on a hard kill, which
# ``verify_audit_integrity`` already detects. The default is per-event
# fsync because durability beats throughput for the safety log.
_SEQ_LOCK = threading.Lock()
_SEQ_COUNTERS: dict[str, int] = {}


class _ProcessAuditState:
    """Container for module-level mutable flags.

    Same rationale as ``mesh/security.py::_ProcessSecurityState``: we
    keep the one-shot ``loaded`` flag on an instance attribute so static
    analysers see a normal attribute read+write rather than a
    ``global`` declaration on a module-level scalar (which CodeQL's
    "unused global variable" rule mis-classifies — alert #222).

    R4-2: ``psk_was_present`` snapshots whether ``STRANDS_MESH_AUDIT_PSK``
    was set at the time the first audit record was signed. Subsequent
    records compare to this snapshot — if the PSK gets unset mid-run,
    ``_sign_record`` logs an ERROR and the record is rejected. This
    closes the "process clears its env briefly to write unsigned
    forgeries, then re-sets the PSK" attack documented at
    review feedback round 4 / R4-2.
    """

    __slots__ = ("seq_loaded", "psk_was_present")

    def __init__(self) -> None:
        self.seq_loaded: bool = False
        # ``None`` = not yet observed; ``True``/``False`` = first-call
        # snapshot.
        self.psk_was_present: bool | None = None


_AUDIT_STATE = _ProcessAuditState()

__all__ = [
    "AuditPSKDegradedError",
    "audit_log_path",
    "log_safety_event",
    "read_audit_log",
    "verify_audit_integrity",
]


def _audit_psk() -> bytes | None:
    """Return the audit-log PSK as bytes, or None when not configured."""
    psk = os.getenv("STRANDS_MESH_AUDIT_PSK")
    if not psk:
        return None
    return psk.encode("utf-8")


def _resolve_log_max_bytes() -> int:
    """Read STRANDS_MESH_AUDIT_MAX_BYTES with a hard upper cap.

    Reject obviously-broken values (negative, zero, larger than 10 GiB)
    and fall back to the default. The hard cap exists so a typo or
    misguided "disable rotation" attempt cannot turn the audit log
    back into an unbounded growth surface.
    """
    raw = os.getenv("STRANDS_MESH_AUDIT_MAX_BYTES")
    if not raw:
        return _DEFAULT_LOG_MAX_BYTES
    try:
        v = int(raw)
    except ValueError:
        logger.warning("[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%r invalid — using default", raw)
        return _DEFAULT_LOG_MAX_BYTES
    if v <= 0:
        return _DEFAULT_LOG_MAX_BYTES
    if v > _LOG_MAX_BYTES_CAP:
        logger.warning(
            "[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%d exceeds hard cap %d — clamping",
            v,
            _LOG_MAX_BYTES_CAP,
        )
        return _LOG_MAX_BYTES_CAP
    return v


def _resolve_log_max_files() -> int:
    """Read STRANDS_MESH_AUDIT_MAX_FILES with a hard upper cap."""
    raw = os.getenv("STRANDS_MESH_AUDIT_MAX_FILES")
    if not raw:
        return _DEFAULT_LOG_MAX_FILES
    try:
        v = int(raw)
    except ValueError:
        return _DEFAULT_LOG_MAX_FILES
    if v < 1:
        return 1
    if v > _LOG_MAX_FILES_CAP:
        return _LOG_MAX_FILES_CAP
    return v


def _rotate_log_if_needed(path: Path, current_size: int) -> None:
    """Rotate the audit log when it exceeds the configured size cap.

    Caller MUST hold :data:`_WRITE_LOCK` so two threads don't both
    rotate. We rename ``mesh_audit.jsonl`` -> ``mesh_audit.jsonl.1``,
    cascading older rotations up the chain, and discarding any
    rotation past ``max_files``.

    Rotation keeps the audit history within bounded disk usage
    (default: 100 MiB x 5 files = 500 MiB). Older records are
    discarded — operators who need long-term retention should ship
    rotated files to durable storage out-of-band.

    Defence: also reject rotation if ``path`` is a symlink (paranoid
    repeat of the F2 check; an attacker who races us between the
    write check and rotation could otherwise redirect the rotated
    name).
    """
    max_bytes = _resolve_log_max_bytes()
    if current_size < max_bytes:
        return
    if path.is_symlink():
        logger.warning("[audit] refusing to rotate symlinked audit log %s", path)
        return

    max_files = _resolve_log_max_files()
    # Cascade .{n} -> .{n+1}, dropping the oldest.
    for n in range(max_files - 1, 0, -1):
        src_p = path.with_suffix(path.suffix + f".{n}")
        dst_p = path.with_suffix(path.suffix + f".{n + 1}")
        if src_p.exists():
            try:
                if n + 1 > max_files - 1:
                    # Discard files past the max. Use os.unlink so a
                    # symlink at this position cannot redirect a delete.
                    if src_p.is_symlink():
                        logger.warning("[audit] discarding symlinked rotated log %s", src_p)
                        src_p.unlink(missing_ok=True)
                        continue
                    src_p.unlink(missing_ok=True)
                else:
                    os.replace(src_p, dst_p)
            except OSError as exc:
                logger.warning("[audit] rotation cascade failed at %s: %s", src_p, exc)
    # Finally, rename the active log to .1 and let the next write
    # create a fresh empty file via O_CREAT.
    try:
        os.replace(path, path.with_suffix(path.suffix + ".1"))
        logger.info("[audit] rotated %s (size=%d bytes)", path, current_size)
    except OSError as exc:
        logger.warning("[audit] could not rotate %s: %s", path, exc)


def _seq_sidecar_path() -> Path:
    """Return the location of the sequence-counter sidecar file."""
    return audit_log_path().parent / "mesh_audit.seq.json"


def _seq_lockfile_path() -> Path:
    """Path to the cross-process lockfile guarding the seq sidecar.

    R5-3: two processes that host the same peer_id (multi-Mesh test
    harness, supervised restart racing the parent, fleet duplicate)
    could otherwise both load the sidecar at seq=N, increment in
    memory to N+1 and N+2 independently, persist whichever arrives
    last, and roll the counter back. We use a separate lockfile
    rather than ``flock``-ing the sidecar itself so the rename in
    ``_persist_seq_counters`` (which atomically replaces the inode)
    cannot strand the lock.
    """
    return audit_log_path().parent / "mesh_audit.seq.lock"


@contextlib.contextmanager
def _seq_flock() -> Iterator[None]:
    """Hold an exclusive flock on the seq lockfile for the block.

    Caller MUST already hold :data:`_SEQ_LOCK` (intra-process). Lock
    ordering: intra-process first, inter-process second. The lock is
    released on context exit even if the caller raises.

    On Windows ``fcntl`` is unavailable; we fall back to in-process
    locking only and document the cross-process limitation in the
    module docstring. POSIX deployments (the supported surface) get
    the full guarantee.
    """
    if not _HAS_FCNTL:
        yield
        return
    lockfile = _seq_lockfile_path()
    try:
        lockfile.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("[audit] cannot create seq lockfile dir: %s", exc)
        yield
        return
    try:
        fd = os.open(str(lockfile), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        logger.debug("[audit] cannot open seq lockfile: %s", exc)
        yield
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            # Best-effort unlock; on close the kernel releases anyway.
            pass
        try:
            os.close(fd)
        except OSError:
            # Already closed; no leak.
            pass


def _load_seq_counters() -> None:
    """Restore ``_SEQ_COUNTERS`` from the sidecar file. Idempotent.

    Caller MUST hold :data:`_SEQ_LOCK`. Stores the one-shot "loaded"
    flag on :data:`_AUDIT_STATE` so static analysers don't trip on a
    bare ``global`` for a module-level scalar.
    """
    if _AUDIT_STATE.seq_loaded:
        return
    sidecar = _seq_sidecar_path()
    try:
        if sidecar.exists():
            with open(sidecar, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(key, str) and isinstance(value, int) and value >= 0:
                        # Only restore if our in-memory value is lower —
                        # never roll a counter backwards even if the file
                        # somehow has a stale value.
                        if value > _SEQ_COUNTERS.get(key, 0):
                            _SEQ_COUNTERS[key] = value
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[audit] could not load seq sidecar %s: %s", sidecar, exc)
    _AUDIT_STATE.seq_loaded = True


def _persist_seq_counters() -> None:
    """Write ``_SEQ_COUNTERS`` to the sidecar file. Fail-soft.

    Caller MUST hold :data:`_SEQ_LOCK`.

    Defence (Phase-4 / Cycle 4 / F3): if the sidecar is a symlink,
    refuse to write. Same threat model as the audit log itself —
    attacker swaps the file with a symlink to redirect counter state
    or null-route it. The atomic ``tmp + os.replace`` already prevents
    half-written sidecars; this adds protection against tamper.
    """
    sidecar = _seq_sidecar_path()
    if sidecar.is_symlink():
        logger.warning(
            "[audit] refusing to persist seq sidecar at %s: it is a SYMLINK "
            "(target: %r). Counter persistence will fail-soft.",
            sidecar,
            os.readlink(sidecar),
        )
        return
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename so a crash mid-write cannot
        # leave a half-formed sidecar that fails to parse on next load.
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        # Open the tmp file with O_NOFOLLOW too, so a TOCTOU between
        # the is_symlink check above and this open is foiled.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp, flags | nofollow, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(_SEQ_COUNTERS, fh, sort_keys=True, separators=(",", ":"))
                fh.flush()
                # R4-3: fsync the temp fd before rename so a power-loss
                # cannot leave the audit log ahead of the sidecar. After
                # restart, ``_load_seq_counters`` would otherwise pick up
                # stale counters and the next event would write a duplicate
                # seq value, defeating per-peer adjacency in
                # ``verify_audit_integrity``.
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Best-effort on filesystems that reject fsync; the
                    # data is in the kernel page cache and durability is
                    # weaker than ideal but the safety code path stays
                    # alive (audit persistence is fail-soft by contract).
                    pass
        except Exception:
            try:
                os.close(fd)
            except OSError:
                # Already closed by fdopen's context manager exit (or
                # never opened). The raise below propagates the original
                # error; this cleanup branch only matters on the rare
                # crash path where fdopen itself raised.
                pass
            raise
        os.replace(tmp, sidecar)
        # R4-3: fsync the parent directory so the rename is durable
        # too. POSIX-only — Windows treats os.fsync on a directory fd
        # as undefined behaviour. Skip the dir fsync there; the rename
        # is atomic on NTFS so the visible-state ordering still holds.
        if os.name == "posix":
            try:
                dir_fd = os.open(str(sidecar.parent), os.O_RDONLY)
            except OSError:
                # Best-effort; if the parent is unreadable the rename
                # still happened and we just lose dir-level durability.
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    # Some filesystems reject directory fsync; the
                    # rename is still on disk in the page cache.
                    pass
                finally:
                    try:
                        os.close(dir_fd)
                    except OSError:
                        # Best-effort close of a read-only dir fd; if
                        # the dir was unmounted between open and close
                        # this can race but no leak occurs because the
                        # process is exiting.
                        pass
        try:
            os.chmod(sidecar, 0o600)
        except OSError:
            # chmod is best-effort: filesystems that don't honour POSIX
            # permissions (FAT32, NFS without uid map, mounted volumes
            # under restricted mount options) silently fail this call,
            # but the sidecar itself is still written and readable. We
            # would rather have a working audit log without 0o600 than
            # crash safety persistence over a chmod failure.
            pass
    except OSError as exc:
        logger.warning("[audit] could not persist seq sidecar %s: %s", sidecar, exc)


def _next_seq(peer_id: str) -> int:
    """Return the next monotonic sequence number for *peer_id*.

    R5-3: the load+increment+persist sequence runs under TWO locks:

    * :data:`_SEQ_LOCK` (intra-process) so multiple Mesh instances in
      one process don't interleave increments.
    * an ``fcntl.flock`` on the sidecar lockfile (inter-process) so
      two processes that share the same audit dir cannot both load
      seq=N and persist different increments — which would roll the
      counter back. Inside the flock we **re-read** the sidecar so
      our in-memory ``_SEQ_COUNTERS`` cache is reconciled with whatever
      a peer process has written since our last increment.

    Lock ordering: intra-process first, inter-process second. Always.
    """
    with _SEQ_LOCK:
        with _seq_flock():
            # Re-read the sidecar inside the flock so a peer process's
            # increments are merged into our in-memory cache before
            # we decide our next value.
            _AUDIT_STATE.seq_loaded = False
            _load_seq_counters()
            next_value = _SEQ_COUNTERS.get(peer_id, 0) + 1
            _SEQ_COUNTERS[peer_id] = next_value
            _persist_seq_counters()
            return next_value


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    """Stable byte encoding for HMAC. Excludes the ``sig`` field."""
    return json.dumps(
        {k: v for k, v in record.items() if k != "sig"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class AuditPSKDegradedError(RuntimeError):
    """Raised when STRANDS_MESH_AUDIT_PSK was set at first write but is
    no longer set at a subsequent write.

    Round-4 / R4-2: a process that briefly clears its env to write a run
    of unsigned forgeries — then re-sets the PSK — would otherwise yield
    records that ``verify_audit_integrity`` reports as ``missing_sig``
    while ``ok`` (the boolean reader-helpers check) stays True. We snap
    PSK presence on the first signed record and refuse to write further
    records under a downgraded configuration.
    """


def _sign_record(record: dict[str, Any]) -> str | None:
    """Compute the per-record HMAC signature, or ``None`` when no PSK
    is configured.

    Round-4 / R4-2: snapshot the PSK presence on the first call. If a
    subsequent call sees the PSK has gone missing (env unset mid-run),
    raise ``AuditPSKDegradedError`` so the audit log cannot silently
    degrade to unsigned. The caller is the safety code path; we let the
    error propagate to ``log_safety_event`` which logs at ERROR and
    swallows it (audit failures must not crash the safety path), but
    we DO refuse the unsigned write.
    """
    psk = _audit_psk()
    snapshot = _AUDIT_STATE.psk_was_present
    if snapshot is None:
        # First record this process — record the observed state.
        _AUDIT_STATE.psk_was_present = psk is not None
    elif snapshot is True and psk is None:
        # Was signed; now unsigned. Refuse.
        raise AuditPSKDegradedError(
            "STRANDS_MESH_AUDIT_PSK was set when the audit log first "
            "started signing this run, but is now unset. Refusing to "
            "write an unsigned record (would silently degrade audit "
            "integrity). Restore the PSK or restart the process to "
            "transition to unsigned mode deliberately."
        )
    if psk is None:
        return None
    return hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()


def audit_log_path() -> Path:
    """Return the resolved path of the audit log file.

    Honours ``STRANDS_MESH_AUDIT_DIR`` (override) or falls back to
    ``~/.strands_robots``.  Does not create the directory.
    """
    override = os.getenv("STRANDS_MESH_AUDIT_DIR")
    base = Path(override).expanduser() if override else _DEFAULT_DIR
    return base / _LOG_FILE_NAME


def _ensure_paths(path: Path) -> None:
    """Make sure the parent directory exists (mode 0o700) and the file
    exists with mode 0o600.

    Re-applies permissions on every call so a fresh deploy or a manual
    ``touch`` cannot leave the file world-readable by accident.

    Defence: if the audit log path is a SYMLINK (potentially pointing
    to attacker-controlled territory like ``/dev/null`` or another
    process's file), refuse to operate. The audit log must always be
    a real regular file at the canonical location. See
    review feedback round 4 (symlink-swap defence).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as exc:  # pragma: no cover — best-effort on exotic FS
        logger.debug("[audit] could not chmod %s: %s", parent, exc)

    # Symlink check on the audit log itself. We use lstat so we don't
    # follow the link; any symlink — even one pointing to a legitimate
    # location — is treated as tampering and rejected. Operators who
    # legitimately want the log to live under a different path use
    # STRANDS_MESH_AUDIT_DIR; symlinks are never the answer.
    try:
        if path.is_symlink():
            raise OSError(
                f"refusing to use audit log at {path}: it is a SYMLINK "
                f"(target: {os.readlink(path)!r}). This may indicate "
                f"tampering. Set STRANDS_MESH_AUDIT_DIR if you need to "
                f"relocate the log."
            )
    except OSError:
        # Re-raise the symlink check; an unrelated lstat failure
        # (permission denied on parent, etc.) is handled by the
        # subsequent open() call.
        if path.is_symlink():
            raise

    if not path.exists():
        # Create the file empty so we can chmod it before writing data.
        path.touch()

    try:
        os.chmod(path, 0o600)
    except OSError as exc:  # pragma: no cover
        logger.debug("[audit] could not chmod %s: %s", path, exc)


def log_safety_event(event_type: str, peer_id: str, payload: dict[str, Any]) -> None:
    """Append a single safety event to the audit log.

    Args:
        event_type: Short, lowercase event identifier
            (e.g. ``"emergency_stop"``).
        peer_id: The mesh peer that originated the event.
        payload: Event-specific fields.  Must be JSON-serialisable.

    Raises:
        Nothing — write errors are logged at WARNING and swallowed because
        an audit-log failure must never propagate up into the safety code
        path that called this function.
    """
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": event_type,
        "peer_id": peer_id,
        "payload": payload,
        "seq": _next_seq(peer_id),
    }
    try:
        sig = _sign_record(record)
    except AuditPSKDegradedError as exc:
        # R4-2: STRANDS_MESH_AUDIT_PSK was set at start of run but is
        # now unset. We refuse to write an unsigned record because the
        # default reader-helper path (`verify_audit_integrity` → `ok`)
        # would otherwise miss the downgrade. Log loud, swallow — audit
        # failures must not crash the safety code path.
        logger.error("[audit] %s — record dropped: %s", exc, record)
        return
    if sig is not None:
        record["sig"] = sig

    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    path = audit_log_path()

    with _WRITE_LOCK:
        try:
            _ensure_paths(path)
            # Phase-4 / E1: rotate BEFORE writing if the active log
            # has grown past the size cap. Rotation is bounded so an
            # attacker flooding events cannot exhaust disk; only the
            # last (max_files * max_bytes) of audit history is kept.
            try:
                cur_size = path.stat().st_size if path.exists() else 0
            except OSError:
                cur_size = 0
            if cur_size > 0:
                _rotate_log_if_needed(path, cur_size)
            # Open with O_NOFOLLOW (POSIX) to defeat a symlink-swap
            # race between _ensure_paths and this open(). On a
            # symlink target the open() raises ELOOP and we reject the
            # write — matching the static check in _ensure_paths.
            #
            # Fall back to plain open() if O_NOFOLLOW is unavailable
            # (Windows). On those platforms the static is_symlink
            # check in _ensure_paths is the only defence; we accept
            # that as residual risk because the supported deployment
            # surface is POSIX (Linux + macOS).
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags | nofollow, 0o600)
            except OSError as oe:
                # ELOOP under O_NOFOLLOW = symlink → retry without it
                # is NOT what we want; just re-raise.
                raise oe
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())  # durable write before returning
                    except OSError:
                        # best-effort on filesystems that reject fsync;
                        # the data is in the kernel page cache and
                        # we'd rather lose durability than crash safety.
                        pass
            except Exception:
                # Make sure fd is closed if fdopen raises.
                try:
                    os.close(fd)
                except OSError:
                    # Already closed by fdopen context-manager exit.
                    # Nothing to do; the original error propagates via
                    # the raise below.
                    pass
                raise
        except OSError as exc:
            logger.warning("[audit] failed to write %s: %s", path, exc)


def read_audit_log(since: float | None = None) -> list[dict[str, Any]]:
    """Read the audit log and return parsed event records.

    Args:
        since: Optional UNIX timestamp.  When provided, only records with
            ``ts >= since`` are returned.

    Returns:
        List of event dicts in the order they were written.  Returns an
        empty list if the log file does not exist.
    """
    path = audit_log_path()
    if not path.exists():
        return []

    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    # Forward-compatible: skip malformed lines silently so a
                    # newer writer's extension can't break a reader on this
                    # version.
                    continue
                if since is not None:
                    ts = record.get("ts")
                    if not isinstance(ts, (int, float)) or ts < since:
                        continue
                out.append(record)
    except OSError as exc:  # pragma: no cover — best-effort read
        logger.debug("[audit] failed to read %s: %s", path, exc)

    return out


def verify_audit_integrity(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Walk the audit log and report tamper / truncation evidence.

    Args:
        records: Optional pre-loaded records list. When None, the current
            audit file is read fresh.

    Returns:
        Dict with keys::

            {
                "total":        <int>,   # records examined
                "signed":       <int>,   # records with a sig field
                "verified":     <int>,   # records whose sig validated
                "bad_signature":<int>,   # records whose sig failed
                "missing_sig":  <int>,   # signed log expected but missing
                "psk_present":  <bool>,  # whether STRANDS_MESH_AUDIT_PSK was set
                "sequence_gaps":[(prev_seq, this_seq), ...],
                "ok":           <bool>,  # True iff bad_signature == 0 and
                                         # sequence_gaps == [].
            }
    """
    if records is None:
        records = read_audit_log()

    psk = _audit_psk()
    psk_present = psk is not None

    total = len(records)
    signed = 0
    verified = 0
    bad_signature = 0
    missing_sig = 0
    gaps: list[tuple[int, int]] = []

    # Track sequence per peer_id — each process has its own counter so the
    # only stream where consecutive seq values must be adjacent is a single
    # peer's contributions.
    last_seq_by_peer: dict[str, int] = {}

    for record in records:
        sig = record.get("sig")
        seq = record.get("seq")
        peer = record.get("peer_id", "")

        record_is_bad = False
        if sig is not None:
            signed += 1
            if psk is None:
                # Cannot verify without PSK; record as unverifiable.
                continue
            expected = hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, expected):
                verified += 1
            else:
                bad_signature += 1
                record_is_bad = True
        else:
            if psk_present:
                missing_sig += 1

        # Only advance the per-peer cursor on records we actually trust.
        # If we let a tampered record update last_seq_by_peer, an attacker
        # who edits a record's claimed seq value could hide a real gap
        # caused by deleting subsequent records — the cursor would jump
        # to the forged value and the next legit record would look adjacent.
        if record_is_bad:
            continue

        if isinstance(seq, int) and isinstance(peer, str):
            prev = last_seq_by_peer.get(peer)
            if prev is not None and seq != prev + 1:
                gaps.append((prev, seq))
            last_seq_by_peer[peer] = seq

    return {
        "total": total,
        "signed": signed,
        "verified": verified,
        "bad_signature": bad_signature,
        "missing_sig": missing_sig,
        "psk_present": psk_present,
        "sequence_gaps": gaps,
        # R4-2: when a PSK is configured at verification time, an
        # unsigned record (missing_sig > 0) is treated as a failure —
        # otherwise an attacker who briefly cleared the env mid-run
        # could write a stretch of unsigned forgeries and the
        # ``ok=True`` reader path would not flag them.
        "ok": bad_signature == 0 and not gaps and not (psk_present and missing_sig > 0),
    }
