"""One on-disk store for the detached sessions the lerobot tools start.

``lerobot_teleoperate`` and ``lerobot_train`` both run their work in a detached
child and both record it in one file under a directory they share on purpose, so
``list`` shows every robot session at once. That file is the only place a child's
pid is written down - ``status`` and ``stop`` find the process only through the
record - so losing a record leaves the child running, holding a GPU or driving an
arm, with no supported way left to stop it. Hence:

* **A read never writes.** Both tools read the whole file, so a prune on the read
  path deletes the other tool's records. Presence is not the running claim -
  ``list`` and ``status`` derive that from
  :func:`~strands_robots.tools._process_stop.session_is_running` when asked - so
  retaining a record never over-reports it.
* **One prune policy, applied only while writing.** :meth:`SessionManager.add_session`
  already holds the lock and is already rewriting the document, so it reaps there.
  It drops only a record naming a usable pid that is provably gone; a pid that
  exists but cannot be inspected, and a ``pid`` field that is not a process id at
  all, are kept, because neither is evidence the run ended.
* **The load-modify-write is locked**, on a lock file beside the store rather than
  the store itself, which :func:`os.replace` swaps out from under a lock.

The directory is created on the first write, not at import: importing a tool must
not write to whatever directory the process happens to be in.
"""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil

from strands_robots.tools._process_stop import (
    recorded_pid,
    session_is_running,
    store_sessions,
)

logger = logging.getLogger(__name__)

#: Directory holding the shared session store and each session's log file.
#: Resolved from the working directory at import but *not* created - see
#: :func:`session_dir`. Rebind this name to redirect the store.
SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"

#: The store both tools read and write, inside :func:`session_dir`.
SESSIONS_FILENAME = "active_sessions.json"


def session_dir() -> Path:
    """The session directory as :data:`SESSION_DIR` is bound now.

    Not created here - a read must not write; :func:`ensure_session_dir` is the
    write path.
    """
    return SESSION_DIR


def ensure_session_dir() -> Path:
    """Create the session directory if needed and return it.

    Called when something is about to be written - a record, or a session's log
    file - rather than at import.

    Raises:
        OSError: The directory could not be created.
    """
    directory = session_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class SessionManager:
    """The shared store of detached tool sessions, keyed by session name.

    Both lerobot session tools use this one class over one file; the module
    docstring states the store's contract.
    """

    def __init__(self, sessions_dir: Path | None = None) -> None:
        """Point the manager at a session store.

        Args:
            sessions_dir: Directory holding the store. Defaults to
                :func:`session_dir`, which is what both tools use.
        """
        self.sessions_file = (session_dir() if sessions_dir is None else Path(sessions_dir)) / SESSIONS_FILENAME

    @property
    def lock_file(self) -> Path:
        """The lock guarding load-modify-write on :attr:`sessions_file`.

        A separate file: the store is committed with :func:`os.replace`, so a lock
        taken on the store guards the inode the commit replaces and a second
        writer locking the new inode would not be excluded.
        """
        return self.sessions_file.with_name(self.sessions_file.name + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive lock across a load-modify-write of the store.

        Both tools change one record of a whole document they first read, so
        without this the later writer stores a map built from a state the earlier
        one already replaced, and the earlier record is lost.

        A lock that cannot be taken is reported and the write proceeds: the store
        sits on whatever filesystem the working directory is on, and one without
        :func:`fcntl.flock` must not make sessions unrecordable.
        """
        ensure_session_dir()
        try:
            handle = open(self.lock_file, "a+", encoding="utf-8")  # noqa: SIM115 - closed in the finally below
        except OSError as e:
            logger.error(f"Error locking sessions: {e}")
            yield
            return
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                logger.warning(
                    "Could not lock the session store at %s (%s); writing without it, "
                    "so a concurrent write may drop a record",
                    self.lock_file,
                    e,
                )
                yield
            else:
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _read(self) -> dict[str, Any]:
        """Return every stored record, dropping none and writing nothing.

        Returns:
            Every stored record, keyed by session name. A store that cannot be
            read degrades to empty rather than raising - including one carrying
            bytes this encoding does not describe, read as U+FFFD so a record
            damaged outside its pid still names its process, a pid being ASCII.
        """
        if not self.sessions_file.exists():
            return {}
        try:
            # A decode policy that cannot raise. The handler below names the two
            # failures this store was expected to have - it is gone, or it is not
            # JSON - and an undecodable byte is neither: ``UnicodeDecodeError`` is
            # a ``ValueError``, so it would pass both clauses and abort the tool
            # action that asked.
            with open(self.sessions_file, encoding="utf-8", errors="replace") as f:
                sessions: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sessions: {e}")
            return {}
        if not isinstance(sessions, dict):
            logger.error(f"Error loading sessions: {self.sessions_file} does not hold a session map")
            return {}
        return sessions

    def _load_sessions(self) -> dict[str, Any]:
        """Load every stored record, reporting any it could not inspect.

        Nothing is dropped and nothing is written: both tools share this read
        path, so a prune here would delete the other tool's records. Finished
        records are reaped by :meth:`add_session`.

        Returns:
            Every stored record, keyed by session name.
        """
        sessions = self._read()
        self._report_uninspectable(sessions)
        return sessions

    def _report_uninspectable(self, sessions: Mapping[str, Any]) -> None:
        """Warn for each session this store holds but cannot inspect.

        Two records read that way, and :meth:`add_session` prunes neither: one
        names a pid that exists and may not be read (a session started under
        ``sudo`` for serial-port access, later listed as the invoking user), the
        other names no pid at all. The warning is the operator's only clue that
        ``list`` and ``status`` are reporting on a run they could not check.

        Args:
            sessions: The loaded records. Inspected only; never modified.
        """
        for name, info in sessions.items():
            pid = recorded_pid(info)
            if pid is None:
                if info.get("pid") is not None:
                    logger.warning(
                        "Session '%s' records a %s as its PID, which is not a process id; "
                        "its record is kept, but the run can only be stopped by hand",
                        name,
                        type(info.get("pid")).__name__,
                    )
                continue
            if not psutil.pid_exists(pid):
                # A finished run. Reported by nothing: it is reaped by the next
                # write, and until then ``list`` and ``status`` derive "Stopped"
                # from the pid themselves.
                continue
            try:
                # Called for what it raises, not for what it returns. Existence is
                # already established, so this probe exists only to surface a
                # denial - the same denial the identity comparison would meet,
                # since both have to read the process rather than only signal it.
                psutil.Process(pid).is_running()
            except psutil.NoSuchProcess:
                # Reaped between the two probes: the same finished run as a pid
                # that was already gone.
                pass
            except psutil.AccessDenied:
                logger.warning(
                    "Session '%s' (PID %s) exists but cannot be inspected; "
                    "keeping its record so the session stays stoppable",
                    name,
                    pid,
                )

    def _save_sessions(self, sessions: Mapping[str, Any]) -> None:
        """Store the session map in full, or leave the stored one untouched.

        :func:`~strands_robots.tools._process_stop.store_sessions` owns the
        sequence, because losing this store is what makes a live session
        unstoppable and both session tools write this one file.
        """
        try:
            ensure_session_dir()
            store_sessions(self.sessions_file, sessions)
        except OSError as e:
            logger.error(f"Error saving sessions: {e}")

    @staticmethod
    def _is_finished(info: Mapping[str, Any]) -> bool:
        """Whether a record's process is *provably* gone, and so may be dropped.

        Args:
            info: A session record.

        Returns:
            ``True`` only when the record names a usable pid that no longer holds
            the process it was written for. A record naming no usable pid is not
            provably anything, so it is kept; and
            :func:`~strands_robots.tools._process_stop.session_is_running`
            answers a refused inspection with ``True``, so a session this user may
            not read is kept too.
        """
        return recorded_pid(info) is not None and not session_is_running(info)

    def add_session(self, name: str, info: dict[str, Any]) -> None:
        """Record a session under ``name``, reaping the runs that have finished.

        The store's only prune happens here: the one place already holding the
        lock and already rewriting the whole document, so it costs no extra write
        and cannot be triggered by another tool merely reading. Only provably
        finished records are dropped (:meth:`_is_finished`).

        Args:
            name: Session name. An existing record under this name is replaced.
            info: The record to store, typically carrying ``pid``, ``log_file``
                and the identity
                :data:`~strands_robots.tools._process_stop.PID_STARTED_SINCE_BOOT`
                names.
        """
        with self._locked():
            sessions = self._read()
            self._report_uninspectable(sessions)
            kept = {
                stored: record for stored, record in sessions.items() if stored == name or not self._is_finished(record)
            }
            kept[name] = info
            self._save_sessions(kept)

    def remove_session(self, name: str) -> None:
        """Drop the record stored under ``name``, leaving every other untouched.

        No reaping here: ``stop`` names one session, and a stop of one tool's
        session must not decide the fate of another tool's records.

        Args:
            name: Session name. A name the store does not hold is not an error.
        """
        with self._locked():
            sessions = self._read()
            if name in sessions:
                del sessions[name]
                self._save_sessions(sessions)

    def get_session(self, name: str) -> dict[str, Any] | None:
        """Return the record stored under ``name``, or ``None``.

        Args:
            name: Session name.

        Returns:
            The stored record, unfiltered:
            :func:`~strands_robots.tools._process_stop.session_is_running` answers
            whether its process still runs, when the caller asks.
        """
        return self._load_sessions().get(name)

    def list_sessions(self) -> dict[str, Any]:
        """Return every stored session record.

        Returns:
            Every record, keyed by session name; the caller decides which are
            still running. Nothing is pruned or written, so one tool's listing
            cannot erase another tool's session.
        """
        return self._load_sessions()


__all__ = [
    "SESSIONS_FILENAME",
    "SESSION_DIR",
    "SessionManager",
    "ensure_session_dir",
    "session_dir",
]
