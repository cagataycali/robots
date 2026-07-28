"""Append-only audit log for safety-critical mesh events.

Safety actions on a multi-robot mesh (most importantly emergency stops)
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

* ``ts`` -- UNIX timestamp (float seconds, UTC)
* ``event`` -- short event type, e.g. ``"emergency_stop"``
* ``peer_id`` -- the mesh peer that owned the event
* ``payload`` -- free-form dict with event-specific fields
* ``seq`` -- per-peer monotonic sequence number; gaps within a single
  peer's stream indicate missing events.
* ``sig`` -- HMAC-SHA256 hex over the rest of the record. Present only
  when ``STRANDS_MESH_AUDIT_PSK`` is configured.

Integrity verification
----------------------
When ``STRANDS_MESH_AUDIT_PSK`` is set the writer attaches a per-record
HMAC signature to frustrate post-incident tampering.
:func:`verify_audit_integrity` walks the log and reports broken
signatures (edited or truncated records), sequence gaps (deleted
records), and unsigned records in a log that should be signed. A
mixed signed/unsigned log only arises when separate processes write to
the same audit directory at different times; within a single process
:func:`_sign_record` hard-rejects any PSK transition (see
:class:`AuditPSKDegradedError`).

The PSK lives in env / Secrets Manager, never in the file. A reader that
does not have the PSK can still read events; it just cannot verify them.

The file is opened in append mode for every write so concurrent writers
from multiple threads or processes never overwrite each other; ordering
across processes is best-effort.

Reading
-------
:func:`read_audit_log` parses the file line by line and returns a list of
event dicts.  Lines that fail to parse are skipped (defensive: the audit
log is forward-compatible with future fields).
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

# Audit-log rotation. Without a size cap, an attacker who can publish
# to a peer's safety/event topic can fill the robot's disk by spamming
# events. We cap the active log and rotate to a numbered suffix on
# overflow, keeping a bounded number of copies. Operators tune via
# ``STRANDS_MESH_AUDIT_MAX_BYTES`` and ``STRANDS_MESH_AUDIT_MAX_FILES``.
#
# We deliberately don't use logging.handlers.RotatingFileHandler -- it
# doesn't honour O_NOFOLLOW and would re-open the file via the default
# open() on rollover, defeating the symlink guard. Rotation here uses
# os.rename + os.open(O_NOFOLLOW) consistently.
_DEFAULT_LOG_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_LOG_MAX_FILES: int = 5
_LOG_MAX_BYTES_CAP: int = 10 * 1024 * 1024 * 1024  # 10 GiB hard upper bound
_LOG_MAX_FILES_CAP: int = 100

# Serialise writes inside a single process so two threads can't interleave
# bytes inside one append. Different processes still need filesystem-level
# atomicity (one open(..., "a") write per event).
_WRITE_LOCK = threading.Lock()

# Per-peer monotonic sequence counters. Each peer_id has its own counter
# so the (peer_id, seq) pair is unique within one process AND consecutive
# values within a single peer's stream are adjacent, keeping
# :func:`verify_audit_integrity` gap detection meaningful even in
# processes that host multiple Mesh peers.
#
# Counters persist to a sidecar file (``mesh_audit.seq.json`` next to
# the audit log) so a process restart does NOT reset them -- otherwise a
# compromised process could delete records, restart, and yield a clean
# ``verify_audit_integrity()`` because every peer's seq would start over
# at 1. The sidecar is reloaded inside the cross-process lockfile
# (``mesh_audit.seq.lock``) on every ``_next_seq`` call and rewritten
# before the lock is released, so two processes sharing the same
# ``STRANDS_MESH_AUDIT_DIR`` cannot roll the counter back. Writes are
# fail-soft: a failure logs at WARNING but does not break the safety
# code path.
#
# Cross-process guarantee: POSIX (``fcntl.flock``) only. Windows falls
# back to in-process locking; multiple writer processes on one audit
# dir are not supported there.
#
# Accepted limitation: every event triggers a synchronous double write
# (audit line + fsynced sidecar). The transport-layer rate cap
# (``STRANDS_MESH_CMD_RATE_HZ``) bounds the worst-case write rate, and
# durability beats throughput for the safety log.
_SEQ_LOCK = threading.Lock()
_SEQ_COUNTERS: dict[str, int] = {}

#: Upper bound on a per-peer seq value when seeding ``_SEQ_COUNTERS`` from
#: an external source (sidecar OR audit-log walk). A value above this cap
#: is almost certainly a forged record / corrupted sidecar; capping the
#: seed prevents one bad input from silently denying the legitimate
#: writer the next ~billion seq values.
_MAX_SEED_SEQ: int = 100_000_000

# Guards ``_AUDIT_STATE.psk_fingerprint``, which is read+compared+set on
# every :func:`_sign_record` call. Held around the entire fingerprint
# check so the compare-and-set is atomic; otherwise a thread landing
# between ``_audit_psk()`` and the comparison could observe a stale view
# that defeats the PSK-degrade defence.
_PSK_STATE_LOCK = threading.Lock()


class _ProcessAuditState:
    """Container for module-level mutable flags.

    Kept on instance attributes (rather than ``global`` scalars) so
    static analysers see normal attribute reads/writes.

    ``psk_fingerprint`` snapshots a fingerprint of the
    ``STRANDS_MESH_AUDIT_PSK`` value seen on the first record this
    process writes. Subsequent records compare to this snapshot --
    if the PSK gets unset, set, or rotated to a different value
    mid-run, ``_sign_record`` raises :class:`AuditPSKDegradedError`
    and the record is rejected. This blocks a writer clearing the PSK
    to forge unsigned records, and a mid-run rotation that would leave
    no record-internal signal of which PSK was active when.

    The fingerprint is the first 16 bytes of ``sha256(psk)``; storing
    it never leaks the PSK itself.
    """

    __slots__ = ("seq_loaded", "audit_log_seeded", "psk_fingerprint")

    def __init__(self) -> None:
        self.seq_loaded: bool = False
        # Once-per-process flag for the audit-log fallback walk inside
        # ``_load_seq_counters``. The sidecar path is cheap and runs on
        # every ``_next_seq`` call; the audit-log walk is O(records)
        # and runs only when the sidecar is unusable. Without this flag
        # a degraded sidecar made every safety event re-walk the entire
        # rotation set. Resetting ``seq_loaded`` in ``_next_seq`` does
        # not clear this flag, so the walk does not repeat.
        self.audit_log_seeded: bool = False
        # ``None`` = not yet observed; ``b""`` = first call observed NO
        # PSK; any other bytes = fingerprint sha256(psk)[:16].
        self.psk_fingerprint: bytes | None = None


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

    Broken values fall back to the default; the hard cap ensures a typo
    or "disable rotation" attempt cannot make audit growth unbounded.
    """
    raw = os.getenv("STRANDS_MESH_AUDIT_MAX_BYTES")
    if not raw:
        return _DEFAULT_LOG_MAX_BYTES
    try:
        v = int(raw)
    except ValueError:
        logger.warning("[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%r invalid -- using default", raw)
        return _DEFAULT_LOG_MAX_BYTES
    if v <= 0:
        return _DEFAULT_LOG_MAX_BYTES
    if v > _LOG_MAX_BYTES_CAP:
        logger.warning(
            "[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%d exceeds hard cap %d -- clamping",
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
    cascading older rotations up the chain and discarding any rotation
    past ``max_files``. Records past the retention window are gone --
    operators needing long-term retention should ship rotated files to
    durable storage out-of-band.

    Defence: also reject rotation if ``path`` is a symlink (an attacker
    who races us between the write check and rotation could otherwise
    redirect the rotated name).
    """
    max_bytes = _resolve_log_max_bytes()
    if current_size < max_bytes:
        return
    if path.is_symlink():
        logger.warning("[audit] refusing to rotate symlinked audit log %s", path)
        return

    max_files = _resolve_log_max_files()
    # Cascade .{n} -> .{n+1} for n in [max_files ... 1]: a leftover file
    # at .{max_files} is unlinked, then lower suffixes shift up by one.
    for n in range(max_files, 0, -1):
        src_p = path.with_suffix(path.suffix + f".{n}")
        dst_p = path.with_suffix(path.suffix + f".{n + 1}")
        if src_p.exists():
            try:
                if n + 1 > max_files:
                    # Discard files past the cap. Use os.unlink so a
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
    # Finally, rename the active log to.1 and let the next write
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

    Two processes hosting the same peer_id could otherwise both load
    the sidecar at seq=N, increment independently, persist whichever
    arrives last, and roll the counter back. A separate lockfile is
    used rather than ``flock``-ing the sidecar itself so the atomic
    rename in ``_persist_seq_counters`` cannot strand the lock.
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
    # Open with O_NOFOLLOW, matching the audit log, sidecar, and ACL
    # loader: a pre-created symlink at ``mesh_audit.seq.lock`` would
    # otherwise have ``flock`` land on the link target instead of
    # failing closed, breaking the cross-process lock promised above.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(lockfile), os.O_RDWR | os.O_CREAT | nofollow, 0o600)
    except OSError as exc:
        import errno

        if getattr(exc, "errno", None) == errno.ELOOP:
            # Symlinked lockfile: hard-fail rather than silently yield
            # without a lock. log_safety_event catches this exception
            # type and writes a SEQ_LOCK_DEGRADED poison record,
            # preserving forensic visibility.
            raise SeqLockSymlinkError(
                f"audit seq lockfile {lockfile} is a symlink (O_NOFOLLOW rejected); "
                "refusing to silently downgrade cross-process serialisation"
            ) from exc
        # Non-ELOOP errors (e.g. EACCES, ENOSPC) are operational
        # failures, not attacker symlink swaps: degrade to
        # yield-without-lock with a DEBUG log.
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

    Caller MUST hold :data:`_SEQ_LOCK`.

    Opens the sidecar with ``O_NOFOLLOW`` and refuses ``is_symlink``
    paths, mirroring :func:`_persist_seq_counters` -- otherwise a
    symlinked sidecar would let an attacker redirect the counter
    restore to attacker-chosen state (e.g. ``/dev/null`` returning zero
    counters and rolling the cursor back).

    When the sidecar load fails or is rejected as a symlink, seed from
    the audit log by taking max(seq) per peer_id, so garbage written to
    the sidecar cannot reset all sequence counters to 0 on next boot.
    """
    if _AUDIT_STATE.seq_loaded:
        return
    sidecar = _seq_sidecar_path()
    sidecar_loaded = False
    try:
        if sidecar.is_symlink():
            logger.warning(
                "[audit] refusing to load seq sidecar at %s: it is a SYMLINK "
                "(target: %r). Counter restore will fail-soft.",
                sidecar,
                os.readlink(sidecar),
            )
        elif sidecar.exists():
            # O_NOFOLLOW on POSIX defeats a symlink-swap between the
            # is_symlink() check above and this open(). On Windows
            # O_NOFOLLOW is 0 and the static check is the only defence;
            # matches the audit-log open in log_safety_event.
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(str(sidecar), os.O_RDONLY | nofollow)
            with os.fdopen(fd, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                # Track whether any entry was actually merged; an empty
                # dict (or one with no valid entries) must NOT flip
                # sidecar_loaded=True, otherwise the audit-log fallback
                # is skipped and an attacker writing ``{}`` gets every
                # peer's seq reset.
                merged_any = False
                for key, value in payload.items():
                    if not (isinstance(key, str) and isinstance(value, int) and value >= 0):
                        continue
                    # Cap the seed even on the healthy-sidecar path: the
                    # sidecar is not signed, so the cap is the only
                    # defence against a planted value like
                    # ``{"victim_peer_id": 999999999}`` jumping the
                    # counter with no upper bound.
                    if value > _MAX_SEED_SEQ:
                        logger.warning(
                            "[audit] refusing to seed seq counter for %r "
                            "from sidecar value %d (cap=%d, possibly "
                            "tampered sidecar)",
                            key,
                            value,
                            _MAX_SEED_SEQ,
                        )
                        continue
                    # Only restore if our in-memory value is lower --
                    # never roll a counter backwards even if the file
                    # somehow has a stale value.
                    if value > _SEQ_COUNTERS.get(key, 0):
                        _SEQ_COUNTERS[key] = value
                        merged_any = True
                # Either real entries were merged or we fall through to
                # the integrity-checked audit-log seed below.
                sidecar_loaded = merged_any
            else:
                logger.warning(
                    "[audit] sidecar %s parsed as non-dict (%s) -- falling through to audit-log seed",
                    sidecar,
                    type(payload).__name__,
                )
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[audit] could not load seq sidecar %s: %s", sidecar, exc)

    # If the sidecar failed to load (corrupt/symlink/missing), seed
    # from the audit log to prevent fail-open sequence reset.
    #
    # When STRANDS_MESH_AUDIT_PSK is configured, ONLY seed from records
    # whose HMAC ``sig`` verifies -- an attacker who could write to the
    # audit log could otherwise append a forged record with a huge seq
    # and have it become the new floor for that peer on the next
    # restart with a corrupt sidecar. Without a PSK everything is
    # trusted (the dev posture has no integrity gate).
    #
    # ``audit_log_seeded`` gates the O(records) walk to once per
    # process; ``_next_seq`` resets ``seq_loaded`` on every call so the
    # cheap sidecar merge still runs inside the flock, but a degraded
    # sidecar must not re-walk the rotation set on every safety event.
    if not sidecar_loaded and not _AUDIT_STATE.audit_log_seeded:
        try:
            records = read_audit_log()
            psk = _audit_psk()
            verified = 0
            unverified_skipped = 0
            for record in records:
                peer_id = record.get("peer_id")
                seq = record.get("seq")
                if not (isinstance(peer_id, str) and isinstance(seq, int) and seq > 0):
                    continue
                # PSK configured: only trust HMAC-verified records.
                if psk is not None:
                    sig = record.get("sig")
                    if not isinstance(sig, str) or sig in ("PSK_DEGRADED", "SIGN_FAILED"):
                        unverified_skipped += 1
                        continue
                    expected = hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(sig, expected):
                        unverified_skipped += 1
                        continue
                # Cap the seed even without a PSK; :data:`_MAX_SEED_SEQ`
                # is shared with the sidecar path so both seed sources
                # have the same fail-loud-on-tamper posture.
                if seq > _MAX_SEED_SEQ:
                    logger.warning(
                        "[audit] refusing to seed seq counter for %r from "
                        "audit-log seq=%d (cap=%d, possibly forged record)",
                        peer_id,
                        seq,
                        _MAX_SEED_SEQ,
                    )
                    unverified_skipped += 1
                    continue
                if seq > _SEQ_COUNTERS.get(peer_id, 0):
                    _SEQ_COUNTERS[peer_id] = seq
                    verified += 1
            if _SEQ_COUNTERS:
                logger.info(
                    "[audit] seeded %d peer counters from audit log after sidecar load failed (verified=%d, skipped_unverified=%d)",
                    len(_SEQ_COUNTERS),
                    verified,
                    unverified_skipped,
                )
            elif unverified_skipped:
                logger.warning(
                    "[audit] sidecar load failed and %d audit records were unverified -- counters NOT seeded "
                    "(this is the circular-trust defence; an attacker who wrote unsigned forgeries "
                    "to the audit log cannot poison the seq restore when STRANDS_MESH_AUDIT_PSK is set)",
                    unverified_skipped,
                )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as log_exc:
            # Deliberately narrow: disk failures, malformed records,
            # and record-shape mismatches degrade softly; an unexpected
            # exception type is a programmer bug that should surface in
            # tests rather than silently weaken the seed defence.
            logger.warning(
                "[audit] could not seed from audit log after sidecar failure: %s",
                log_exc,
            )
        # Mark the walk done regardless of outcome: while the sidecar
        # stays degraded, the in-memory floor is the best we have and
        # re-walking only burns CPU on the safety code path.
        _AUDIT_STATE.audit_log_seeded = True

    _AUDIT_STATE.seq_loaded = True


def _persist_seq_counters() -> None:
    """Write ``_SEQ_COUNTERS`` to the sidecar file. Fail-soft.

    Caller MUST hold :data:`_SEQ_LOCK`.

    Defence: refuse to write if the sidecar is a symlink (same threat
    model as the audit log -- a symlink swap could redirect or
    null-route counter state). The atomic ``tmp + os.replace`` already
    prevents half-written sidecars.
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
                # fsync the temp fd before rename so a power-loss
                # cannot leave the audit log ahead of the sidecar and
                # a restart write a duplicate seq value, defeating
                # per-peer adjacency in ``verify_audit_integrity``.
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Best-effort on filesystems that reject fsync;
                    # audit persistence is fail-soft by contract.
                    pass
        except OSError:
            # Only reachable when ``os.fdopen`` itself raises before
            # adopting the fd; otherwise the context manager already
            # closed it and ``os.close`` would hit EBADF (suppressed
            # below). The original error propagates via ``raise``.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(tmp, sidecar)
        # fsync the parent directory so the rename is durable too.
        # POSIX-only: Windows treats os.fsync on a directory fd as
        # undefined, and the rename is atomic on NTFS anyway.
        if os.name == "posix":
            try:
                dir_fd = os.open(str(sidecar.parent), os.O_RDONLY)
            except OSError:
                # Best-effort; the rename still happened, we just lose
                # dir-level durability.
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    # Some filesystems reject directory fsync.
                    pass
                finally:
                    try:
                        os.close(dir_fd)
                    except OSError:
                        pass
        try:
            os.chmod(sidecar, 0o600)
        except OSError:
            # Best-effort: filesystems that don't honour POSIX
            # permissions fail this call, but the sidecar is still
            # written. A working audit log without 0o600 beats crashing
            # safety persistence over a chmod failure.
            pass
    except OSError as exc:
        logger.warning("[audit] could not persist seq sidecar %s: %s", sidecar, exc)


def _next_seq(peer_id: str) -> int:
    """Return the next monotonic sequence number for *peer_id*.

    The load+increment+persist sequence runs under TWO locks:
    :data:`_SEQ_LOCK` (intra-process) so multiple Mesh instances in one
    process don't interleave increments, and an ``fcntl.flock`` on the
    sidecar lockfile (inter-process) so two processes sharing an audit
    dir cannot both load seq=N and roll the counter back. Inside the
    flock the sidecar is re-read so ``_SEQ_COUNTERS`` is reconciled
    with peer-process increments.

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


class SeqLockSymlinkError(RuntimeError):
    """Raised when the audit seq lockfile is a symlink.

    A pre-created symlink at ``mesh_audit.seq.lock`` would let
    ``fcntl.flock`` land on the link target rather than fail closed,
    silently downgrading the cross-process serialisation that the
    per-peer monotonic seq guarantee depends on. Hard-fail posture
    matches ``_ensure_paths`` for the audit log itself.
    """


class AuditPSKDegradedError(RuntimeError):
    """Raised when STRANDS_MESH_AUDIT_PSK transitions mid-run.

    A process that briefly clears its env to write a run of unsigned
    forgeries -- then re-sets the PSK -- would otherwise yield records
    that ``verify_audit_integrity`` reports as ``missing_sig`` while
    ``ok`` stays True. PSK presence is snapped on the first record and
    further writes under a degraded configuration are refused.
    """


def _psk_fingerprint(psk: bytes | None) -> bytes:
    """Return ``b""`` if PSK is unset, else the first 16 bytes of
    sha256(psk). Used by :data:`_AUDIT_STATE` to detect mid-run PSK
    transitions (set, unset, OR rotation). One-way: storing the
    fingerprint never leaks the PSK itself.
    """
    if psk is None:
        return b""
    return hashlib.sha256(psk).digest()[:16]


def _sign_record(record: dict[str, Any]) -> str | None:
    """Compute the per-record HMAC signature, or ``None`` when no PSK
    is configured.

    A fingerprint of the PSK is snapped on the first call. If a
    subsequent call sees a different fingerprint (PSK unset, set, or
    rotated), raise ``AuditPSKDegradedError`` so the log cannot
    silently degrade to unsigned, start signing on top of an
    unverifiable unsigned prefix, or switch keys mid-run.

    The caller is the safety code path; the error propagates to
    ``log_safety_event`` which writes a ``sig="PSK_DEGRADED"`` poison
    record and logs at ERROR. Audit failures must not crash the safety
    path, but the unsigned / rotated write IS refused.
    """
    psk = _audit_psk()
    current_fp = _psk_fingerprint(psk)
    # Compare-and-set and comparison run under one lock: two threads
    # racing a PSK rotation could otherwise both read the same
    # pre-rotation snapshot, both pass the comparison, and both write
    # under the new key without one raising AuditPSKDegradedError.
    with _PSK_STATE_LOCK:
        snapshot = _AUDIT_STATE.psk_fingerprint
        if snapshot is None:
            # First record this process -- snap the observed state and
            # treat the current call as matching itself (no transition).
            _AUDIT_STATE.psk_fingerprint = current_fp
            snapshot = current_fp

        if snapshot != current_fp:
            transition_detected = True
        else:
            transition_detected = False
    if transition_detected:
        # PSK transition: set->unset, unset->set, OR rotated value.
        # All three break verifiability symmetrically; refuse.
        if snapshot != b"" and current_fp == b"":
            reason = (
                "STRANDS_MESH_AUDIT_PSK was set when the audit log first "
                "started signing this run, but is now unset. Refusing to "
                "write an unsigned record (would silently degrade audit "
                "integrity). Restore the PSK or restart the process to "
                "transition to unsigned mode deliberately."
            )
        elif snapshot == b"" and current_fp != b"":
            reason = (
                "STRANDS_MESH_AUDIT_PSK was unset when the audit log first "
                "started this run, but is now set. Refusing to start signing "
                "mid-run (would create an unverifiable unsigned prefix that "
                "a forensic walker cannot distinguish from an attacker-forged "
                "forgery window). Restart the process to transition to "
                "signed mode deliberately."
            )
        else:
            # Both non-empty but different: rotated value.
            reason = (
                "STRANDS_MESH_AUDIT_PSK changed value mid-run "
                "(rotation detected via fingerprint). Refusing to "
                "sign records under the new key: a verifier holding "
                "either key would fail signature on the other "
                "segment with no way to attribute records to keys. "
                "Restart the process to rotate the PSK deliberately."
            )
        raise AuditPSKDegradedError(reason)
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
    a real regular file at the canonical location.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as exc:  # pragma: no cover - best-effort on exotic FS
        logger.debug("[audit] could not chmod %s: %s", parent, exc)

    # Symlink check on the audit log itself: the static check gives the
    # eager-fail path, and O_NOFOLLOW on the create below ensures a
    # symlink swap between the two cannot redirect the create.
    if path.is_symlink():
        raise OSError(
            f"refusing to use audit log at {path}: it is a SYMLINK "
            f"(target: {os.readlink(path)!r}). This may indicate "
            f"tampering. Set STRANDS_MESH_AUDIT_DIR if you need to "
            f"relocate the log."
        )

    if not path.exists():
        # Create with O_NOFOLLOW so an attacker who races a symlink in
        # between the is_symlink check above and this open cannot
        # redirect the create (``Path.touch`` would follow symlinks).
        # On Windows where O_NOFOLLOW is 0 the static check above is
        # the only line of defence.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags | nofollow, 0o600)
        except FileExistsError:
            # Another writer created it concurrently; that's fine --
            # they raced ahead of us and the next is_symlink check
            # would catch a swap if one happened.
            pass
        else:
            os.close(fd)

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
        Nothing - write errors are logged at WARNING and swallowed because
        an audit-log failure must never propagate up into the safety code
        path that called this function.

    Failures inside the audit machinery itself are never silent drops:
    the record is written anyway with a poison ``sig`` discriminator
    that a verifier keys on -- ``"PSK_DEGRADED"`` (PSK transitioned
    mid-run), ``"SIGN_FAILED"`` (sign-time error while a PSK is
    configured), ``"SEQ_LOCK_DEGRADED"`` (seq lockfile is a symlink),
    or ``"NEXT_SEQ_DEGRADED"`` (any other seq-counter failure). None of
    these is a valid HMAC, so :func:`verify_audit_integrity` forces
    ``ok=False`` and the gap stays attributable.
    """
    seq_lock_degraded_reason: str | None = None
    next_seq_degraded_reason: str | None = None
    try:
        seq = _next_seq(peer_id)
    except SeqLockSymlinkError as exc:
        # Seq lockfile is a symlink: write a SEQ_LOCK_DEGRADED poison
        # record. seq is unknown -- 0 is the placeholder so the record
        # still serialises.
        logger.error(
            "[audit] SEQ_LOCK_DEGRADED for peer_id=%r: %s -- writing poison record",
            peer_id,
            exc,
        )
        seq = 0
        seq_lock_degraded_reason = str(exc)
    except Exception as exc:  # noqa: BLE001 -- audit log failures MUST be soft per contract
        # Any other _next_seq failure (e.g. OSError on the sidecar, a
        # corrupt counter file): emit a NEXT_SEQ_DEGRADED poison record
        # with seq=0 so the seq-counter gap is classifiable instead of
        # a silent hole.
        logger.error(
            "[audit] NEXT_SEQ_DEGRADED for peer_id=%r: %s -- writing poison record",
            peer_id,
            exc,
        )
        seq = 0
        next_seq_degraded_reason = str(exc)
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": event_type,
        "peer_id": peer_id,
        "payload": payload,
        "seq": seq,
    }
    if next_seq_degraded_reason is not None:
        record["sig"] = "NEXT_SEQ_DEGRADED"
    elif seq_lock_degraded_reason is not None:
        record["sig"] = "SEQ_LOCK_DEGRADED"
        record["seq_lock_degraded"] = seq_lock_degraded_reason
    sig: str | None = None
    if seq_lock_degraded_reason is None and next_seq_degraded_reason is None:
        try:
            sig = _sign_record(record)
        except AuditPSKDegradedError as exc:
            # PSK transitioned mid-run: refuse to forge a signature,
            # but write a PSK_DEGRADED poison record (with a
            # ``psk_degraded`` reason field) instead of dropping.
            logger.error("[audit] %s -- writing poison record (sig=PSK_DEGRADED): %s", exc, record)
            record["sig"] = "PSK_DEGRADED"
            record["psk_degraded"] = str(exc)
        except Exception as sign_exc:  # noqa: BLE001 -- audit must be soft per contract
            # Any other sign-time error must not crash the safety code
            # path. If a PSK is configured, write a SIGN_FAILED poison
            # record so a forensic walker sees the gap as a bad
            # signature -- a plain unsigned record would be invisible to
            # a verifier running without the PSK.
            logger.error(
                "[audit] _sign_record raised %s: %s",
                type(sign_exc).__name__,
                sign_exc,
            )
            if _audit_psk() is not None:
                record["sig"] = "SIGN_FAILED"
                record["sign_error"] = f"{type(sign_exc).__name__}: {sign_exc}"
            # else: no PSK -- the unsigned write is the dev-mode posture.
        else:
            if sig is not None:
                record["sig"] = sig

    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[audit] could not serialise record for peer_id=%r: %s -- record dropped",
            peer_id,
            exc,
        )
        return
    path = audit_log_path()

    with _WRITE_LOCK:
        try:
            _ensure_paths(path)
            # Rotate BEFORE writing if the active log has grown past
            # the size cap, so flooded events cannot exhaust disk.
            try:
                cur_size = path.stat().st_size if path.exists() else 0
            except OSError:
                cur_size = 0
            if cur_size > 0:
                _rotate_log_if_needed(path, cur_size)
            # Open with O_NOFOLLOW (POSIX) to defeat a symlink-swap
            # race between _ensure_paths and this open(): on a symlink
            # the open() raises ELOOP and the write is rejected loudly.
            # Do NOT retry without O_NOFOLLOW. Where O_NOFOLLOW is
            # unavailable (Windows) the static is_symlink check in
            # _ensure_paths is the only defence -- accepted residual
            # risk; the supported deployment surface is POSIX.
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags | nofollow, 0o600)
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())  # durable write before returning
                    except OSError:
                        # Best-effort: rather lose durability on
                        # fsync-rejecting filesystems than crash safety.
                        pass
            except Exception:
                # Make sure fd is closed if fdopen raises; EBADF means
                # the context manager already closed it.
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("[audit] failed to write %s: %s", path, exc)


def _audit_log_files_in_order() -> list[Path]:
    """Return the audit log file set in chronological order.

    Rotated files are named ``mesh_audit.jsonl.N`` where ``.1`` is the
    most recently rotated and higher numbers are older. To iterate in
    chronological order we read the highest-numbered rotation first,
    then descend, then the active log last. When rotation has not
    happened the list is just ``[active]``.

    Returns an empty list when no audit file exists at all.
    """
    active = audit_log_path()
    out: list[Path] = []
    parent = active.parent
    if not parent.is_dir():
        return [active] if active.exists() else []

    # Find every rotated copy (mesh_audit.jsonl.<N>).
    rotations: list[tuple[int, Path]] = []
    for entry in parent.iterdir():
        name = entry.name
        if not name.startswith(active.name + "."):
            continue
        suffix = name[len(active.name) + 1 :]
        if not suffix.isdigit():
            continue
        rotations.append((int(suffix), entry))

    # Rotated suffixes: higher number = older, so sort DESC and prepend
    # the active log at the end.
    rotations.sort(reverse=True)
    out.extend(p for _, p in rotations)
    if active.exists():
        out.append(active)
    return out


def read_audit_log(since: float | None = None) -> list[dict[str, Any]]:
    """Read the audit log and return parsed event records.

    Reads rotated copies (``mesh_audit.jsonl.N``) in chronological
    order before the active log, so verification spans the full
    retained window rather than just the current file.

    Args:
        since: Optional UNIX timestamp.  When provided, only records
            with ``ts >= since`` are returned.

    Returns:
        List of event dicts in chronological order. Returns an empty
        list if no audit file exists.
    """
    # This is the forensic walker AND the seed source for
    # ``_load_seq_counters`` on a corrupt sidecar, so it follows the
    # same is_symlink() + O_NOFOLLOW discipline as every other open in
    # this module: a bare open() would let an attacker who swapped a
    # rotated log file to a symlink redirect the read to
    # attacker-controlled bytes.
    out: list[dict[str, Any]] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for path in _audit_log_files_in_order():
        try:
            if path.is_symlink():
                logger.warning(
                    "[audit] refusing to read %s: it is a SYMLINK (target: %r). Audit log files must be regular files.",
                    path,
                    os.readlink(path),
                )
                continue
            fd = os.open(str(path), os.O_RDONLY | nofollow)
            with os.fdopen(fd, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as parse_exc:
                        # Skipped for forward-compatibility, but with a
                        # DEBUG breadcrumb: a malformed line hides its
                        # seq from the seed walk in _load_seq_counters,
                        # so the seq seed may be incomplete and the
                        # next restart could duplicate a seq value.
                        logger.debug(
                            "[audit] skipping malformed line in %s: %s",
                            path,
                            parse_exc,
                        )
                        continue
                    if since is not None:
                        ts = record.get("ts")
                        if not isinstance(ts, (int, float)) or ts < since:
                            continue
                    out.append(record)
        except OSError as exc:  # pragma: no cover -- best-effort read
            # ELOOP under O_NOFOLLOW is the symlink-raced-after-static-
            # check path; treated as silent skip same as a missing file.
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
                "total":  <int>,  # records examined
                "signed":  <int>,  # records with a sig field
                "verified":  <int>,  # records whose sig validated
                "bad_signature":<int>,  # records whose sig failed
                "missing_sig":  <int>,  # signed log expected but missing
                "unverifiable_signed":<int>, # signed records, verifier lacks PSK
                "psk_present":  <bool>,  # whether STRANDS_MESH_AUDIT_PSK was set
                "sequence_gaps":[(prev_seq, this_seq),...],
                "ok":  <bool>,  # True iff bad_signature == 0 and
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
    unverifiable_signed = 0
    gaps: list[tuple[int, int]] = []

    # Track sequence per peer_id -- each process has its own counter so the
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
                # Verifier lacks the PSK on a signed log: count so
                # ``ok`` fails closed rather than green-lighting a log
                # it cannot actually verify.
                unverifiable_signed += 1
                continue
            expected = hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, expected):
                verified += 1
            else:
                bad_signature += 1
                record_is_bad = True
        else:
            if psk_present:
                # With a PSK, an unsigned record is forged by
                # definition. Mark it bad so the per-peer cursor does
                # not advance past it -- omitting ``sig`` (the natural
                # attack for someone who cannot compute the HMAC) must
                # not jump the cursor and hide deletions from gap
                # detection.
                missing_sig += 1
                record_is_bad = True

        # Only advance the per-peer cursor on records we actually
        # trust: a tampered record updating last_seq_by_peer could hide
        # a real gap by jumping the cursor to a forged seq value.
        if record_is_bad:
            continue

        if isinstance(seq, int) and isinstance(peer, str):
            prev = last_seq_by_peer.get(peer)
            if prev is not None and seq != prev + 1:
                gaps.append((prev, seq))
            # Refuse to roll the cursor backward. A forged record carrying
            # a seq <= prev would let a (forged-low-seq + delete-newer)
            # tamper sequence look adjacent on the next legit record.
            # Keep the highest seq seen for this peer.
            if prev is None or seq > prev:
                last_seq_by_peer[peer] = seq

    return {
        "total": total,
        "signed": signed,
        "verified": verified,
        "bad_signature": bad_signature,
        "missing_sig": missing_sig,
        "unverifiable_signed": unverifiable_signed,
        "psk_present": psk_present,
        "sequence_gaps": gaps,
        # With a PSK, unsigned records (missing_sig > 0) fail ``ok`` --
        # otherwise a stretch of unsigned forgeries written while the
        # env was briefly cleared would go unflagged. Without a PSK on
        # a signed log (unverifiable_signed > 0), fail closed too.
        "ok": (bad_signature == 0 and not gaps and not (psk_present and missing_sig > 0) and unverifiable_signed == 0),
    }
