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

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOG_FILE_NAME = "mesh_audit.jsonl"
_DEFAULT_DIR = Path.home() / ".strands_robots"

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
# threat model. The sidecar is loaded once on the first ``_next_seq``
# call and rewritten under ``_SEQ_LOCK`` after each increment. Writes
# are fail-soft: a write failure logs at WARNING but does not break the
# safety code path.
_SEQ_LOCK = threading.Lock()
_SEQ_COUNTERS: dict[str, int] = {}


class _ProcessAuditState:
    """Container for module-level mutable flags.

    Same rationale as ``mesh/security.py::_ProcessSecurityState``: we
    keep the one-shot ``loaded`` flag on an instance attribute so static
    analysers see a normal attribute read+write rather than a
    ``global`` declaration on a module-level scalar (which CodeQL's
    "unused global variable" rule mis-classifies — alert #222).
    """

    __slots__ = ("seq_loaded",)

    def __init__(self) -> None:
        self.seq_loaded: bool = False


_AUDIT_STATE = _ProcessAuditState()

__all__ = [
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


def _seq_sidecar_path() -> Path:
    """Return the location of the sequence-counter sidecar file."""
    return audit_log_path().parent / "mesh_audit.seq.json"


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
    """
    sidecar = _seq_sidecar_path()
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename so a crash mid-write cannot
        # leave a half-formed sidecar that fails to parse on next load.
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_SEQ_COUNTERS, fh, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, sidecar)
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

    Each peer maintains its own counter under :data:`_SEQ_LOCK`, so two
    peers writing concurrently from the same process produce
    independently-numbered streams that gap-detection can verify.

    Counters are loaded from a sidecar file on the first call per
    process and rewritten after each increment, so a process restart
    does not reset them. Persistence is fail-soft — a write failure is
    logged but does not break the safety code path.
    """
    with _SEQ_LOCK:
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


def _sign_record(record: dict[str, Any]) -> str | None:
    psk = _audit_psk()
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
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as exc:  # pragma: no cover — best-effort on exotic FS
        logger.debug("[audit] could not chmod %s: %s", parent, exc)

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
    sig = _sign_record(record)
    if sig is not None:
        record["sig"] = sig

    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    path = audit_log_path()

    with _WRITE_LOCK:
        try:
            _ensure_paths(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())  # durable write before returning
                except OSError:
                    pass  # best-effort on filesystems that reject fsync
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
        "ok": bad_signature == 0 and not gaps,
    }
