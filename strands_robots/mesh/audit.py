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

The file is opened in append mode for every write so concurrent writers from
multiple threads or processes never overwrite each other; ordering across
processes is best-effort.

Reading
-------
:func:`read_audit_log` parses the file line by line and returns a list of
event dicts.  Lines that fail to parse are silently skipped (defensive: the
audit log is forward-compatible with future fields).
"""

from __future__ import annotations

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

__all__ = ["audit_log_path", "log_safety_event", "read_audit_log"]


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
    record = {
        "ts": time.time(),
        "event": event_type,
        "peer_id": peer_id,
        "payload": payload,
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    path = audit_log_path()

    with _WRITE_LOCK:
        try:
            _ensure_paths(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
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
