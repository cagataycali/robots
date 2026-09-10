"""Confirmation that a signalled session process has actually exited.

A tool that stops a background session sends a signal and then has to answer a
separate question: did the process go away? Sending SIGKILL is not the answer.
The kernel delivers it asynchronously, and a task inside an uninterruptible
wait - a serial ioctl on a teleoperation bus, a stalled CUDA or network call in
a training step - stays in the process table until that wait returns. A stop
that reports success on the strength of having *sent* the signal tells the
caller the arm is released and the GPU is free when neither may be true.

:func:`confirm_exit` is that answer and :func:`unstopped_result` is the report
for when it is not affirmative, shared by every session ``stop`` verb so the
rule is stated once rather than per verb.

The other half of the same question is which process is being asked about. A pid
is not a name: the kernel hands the number back out once the process holding it
exits, so a session record that outlives its run points at whatever holds the
number next. Answering "is it running" from the pid alone therefore reports a
stranger as the session, and the ``stop`` verb that verdict invites signals it.
:func:`session_is_running` is that answer, and it needs the identity
:func:`confirm_exit` already insists on - captured before the question is asked,
because a freshly constructed :class:`psutil.Process` captures the identity it is
being asked to check and so cannot contradict it.

Before either question can be asked there is the number itself, and it arrives
from a JSON file rather than from a caller. :func:`recorded_pid` is the one
reader of it, because converting instead of grading is how a record comes to
name a process it was never written for: ``int(4321.5)`` is a pid the record does
not carry, and ``int(True)`` is pid 1.

The same is true of every other field the record carries, and the one a report
reads is ``start_time``. :func:`session_uptime` is its one reader, for the same
reason: subtracting an ungraded stamp from the clock turns a record that states
no usable start into a duration - the ``0`` an absent key defaults to renders as
the whole epoch, a little under fifty-seven years - and reports it beside a
running flag that is correct.

One store holds every robot session, because the tools that start them are
peers: a detached teleoperation run and a detached training run are two entries
in one file, and either tool's ``list`` shows both. :class:`SessionManager` is
that file's one reader and writer for the same reason :func:`recorded_pid` is
the one reader of a pid. Two readers of one document cannot hold two retention
policies: whichever of them deletes a record deletes it for the other as well,
so the destructive policy is the one that takes effect and the retaining one's
guarantee is not the store's, only its own.

Reading a record presupposes that the file still holds one, which is what
:func:`store_sessions` is for. Every session store here is a whole document: a
verb that starts or stops one session loads every record, changes that one, and
writes them all back. A write that lands partially therefore does not lose the
session being changed, it loses every session the file held - and the load path
reports an unparseable store as *no sessions*, so the loss surfaces as an arm
that is still being driven by a process no verb can name.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# How long to let a process wind itself down after SIGTERM before escalating.
# The session processes this covers flush a dataset shard or a checkpoint on
# the way out, so the grace period is real work, not politeness.
SIGTERM_GRACE_S = 2.0

# How long to wait for SIGKILL to take effect before reporting that it has not.
# A process still present after this is in an uninterruptible wait; more waiting
# does not change the verdict, and the caller needs the verdict.
SIGKILL_CONFIRM_S = 2.0

#: Where every session tool keeps its detached-session records and their logs.
#: One directory, because the store inside it is one document (see
#: :class:`SessionManager`): a second definition of this path is a second
#: independently-redirectable name for one file, which is how two policies came
#: to be applied to it.
SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

#: Session-record key holding the identity of the process the record was written
#: for: how long after boot that process started.
PID_STARTED_SINCE_BOOT = "pid_started_since_boot"

# Tolerance for comparing two reads of that identity. On Linux both reads are the
# same integer tick count the kernel recorded, divided by the tick rate, so they
# are bit-identical and the tolerance is slack. It is not zero because the
# non-Linux fallback below recovers the count by subtracting a boot date, and two
# such subtractions agree only to the float error of the pair. One tick is far
# above that noise, and far below the lifetime of any session, which is the gap a
# reused pid's own start offset differs by.
_IDENTITY_TOLERANCE_S = 0.05

#: Largest process id this platform can be asked about. A pid is a ``pid_t``,
#: a signed 32-bit integer on Linux and macOS, and both ``psutil.pid_exists``
#: and ``os.kill`` raise ``OverflowError`` above it rather than answering - so a
#: larger number in a session record cannot name a process, and grading it out
#: is what keeps that OverflowError from reaching a caller. The real ceiling is
#: ``/proc/sys/kernel/pid_max``, but that is configurable at runtime, and a
#: record written before it was lowered still names its process.
_PID_T_MAX = 2**31 - 1

#: Index of ``starttime`` among the ``/proc/<pid>/stat`` fields that follow the
#: comm field - field 22 of the whole line, counting from 1. Everything up to and
#: including comm is skipped by partitioning on the last ``)``, because a process
#: name may itself contain spaces and parentheses.
_STARTTIME_AFTER_COMM = 19


def _started_since_boot(pid: int) -> float:
    """The kernel's own record of how long after boot ``pid`` started.

    Raises the same two exceptions a :class:`psutil.Process` probe does, so a
    caller can still tell "the process is gone" from "this user may not look" -
    a distinction :func:`session_is_running` turns into opposite verdicts.

    Read from ``/proc/<pid>/stat`` field 22 directly on Linux, rather than as
    ``create_time() - boot_time()``, because that subtraction is not invariant
    under a wall-clock step and the whole point of this identity is that it is.
    ``Process.create_time()`` returns the tick count *plus* a boot date, so the
    boot date has to be subtracted back off, and the two terms are not
    guaranteed to be the same read of it: on psutil at or before 7.0 the process
    side adds a module-level btime cached at import while top-level
    ``boot_time()`` deliberately re-reads ``/proc/stat`` (its own comment: "we
    are not caching this because it is subject to system clock updates"), so a
    step after the cache was populated moves the result by the step size until
    the cache is refreshed. On 7.2 the cache is gone and both terms re-read
    ``/proc/stat``, which narrows the exposure to a step landing between the two
    reads but does not remove it. Field 22 never mentions the wall clock, so
    there is nothing to disagree about on any version.

    Args:
        pid: The pid to identify.

    Returns:
        Seconds between boot and the process's creation.

    Raises:
        psutil.NoSuchProcess: The process is gone.
        psutil.AccessDenied: This user may not inspect it.
    """
    if not sys.platform.startswith("linux"):
        # No procfs. The subtraction is the only route, so the fallback carries
        # the clock-step exposure described above; recorded here rather than in
        # the caller because it is a property of this platform branch alone.
        return psutil.Process(pid).create_time() - psutil.boot_time()
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            after_comm = handle.read().rpartition(b")")[2].split()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise psutil.NoSuchProcess(pid) from exc
    except PermissionError as exc:
        raise psutil.AccessDenied(pid) from exc
    try:
        ticks = float(after_comm[_STARTTIME_AFTER_COMM])
    except (IndexError, ValueError) as exc:
        # A stat line this code cannot parse is not evidence of anything, and
        # guessing would answer the identity question wrongly in one direction
        # or the other. Report it as un-inspectable, which callers already treat
        # as "no evidence".
        raise psutil.AccessDenied(pid) from exc
    return ticks / os.sysconf("SC_CLK_TCK")


def process_started_since_boot(pid: int) -> float | None:
    """How long after boot the process holding ``pid`` started.

    A start offset rather than a creation date, because the two ends of the
    comparison it serves sit in different processes minutes or hours apart: the
    session store is written by the run that started the process and read back by
    a later one. ``create_time()`` is a date - the process's own start ticks plus
    ``/proc/stat``'s ``btime`` - so an NTP correction or a ``date -s`` between the
    write and the read moves it by the size of the step, and a live session would
    read as a stranger. The tick count on its own is a duration and moves for
    nothing, so that is what :func:`_started_since_boot` reads: the kernel's field
    22 directly, rather than a date with the boot time subtracted back off, which
    would reintroduce the wall clock on both sides of a subtraction that is not
    guaranteed to read it once.

    Args:
        pid: The pid to identify.

    Returns:
        Seconds between boot and the process's creation, or ``None`` when the
        process is gone or this user may not inspect it. ``None`` is not evidence
        either way, and callers must not read it as a mismatch.
    """
    try:
        return _started_since_boot(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def recorded_pid(info: Mapping[str, Any]) -> int | None:
    """The process id a session record names, or ``None`` when it names none.

    The read-side counterpart of the pid a session's ``start`` verb wrote, in the
    shape :func:`~strands_robots.utils.declared_count` uses for a count a file
    declares: the store is JSON on disk, so the only two answers a reader can act
    on are the pid itself and the absence of one. A value outside the domain is
    ``None`` - never a nearby number - because ``int()`` of one aims a signal at a
    process the record does not name:

    * ``int(4321.5)`` is ``4321``. ``stop`` signals whatever holds that number,
      which is not the session and need not be related to it at all.
    * ``bool`` is an ``int`` subclass, so ``true`` is pid 1 - ``init`` on Linux.
    * ``int("4321")`` and ``int(" 4321 ")`` both succeed, so a store spelling
      ``psutil.pid_exists`` refuses outright is accepted here instead.
    * ``int(1e400)`` raises ``OverflowError`` and ``int(float("nan"))``
      ``ValueError``. Both are values ``json.load`` produces from a well-formed
      file, and both escape readers whose documented answer for an unusable store
      is "no sessions" rather than an exception.
    * ``0`` and any negative number are refused because ``os.kill`` reads them as
      process *groups*: ``0`` is every process in the caller's own group.
    * A number above :data:`_PID_T_MAX` is refused because the platform cannot be
      asked about it; see that constant.

    Args:
        info: A session record, read for its ``pid`` key.

    Returns:
        The recorded pid, or ``None`` when the record names no usable one. A
        ``pid`` key that is present but is not a process id answers ``None`` too;
        a caller that must tell those apart - to report the difference rather
        than pass it over - compares against the raw value it read, as
        :func:`unusable_pid_result` does.
    """
    pid = info.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    return pid if 0 < pid <= _PID_T_MAX else None


def session_uptime(info: Mapping[str, Any]) -> tuple[float | None, str]:
    """How long the session ``info`` describes has been running, and how to say so.

    The read-side owner of the record's ``start_time``, in the shape
    :func:`recorded_pid` uses for its ``pid``: the value, or the absence of one -
    never a number derived from a stamp that is not one. It answers for both
    consumers of that span in one read of the clock, so a report's ``Uptime``
    line and the ``uptime`` its ``json`` block carries cannot disagree.

    ``start_time`` is a wall-clock stamp, so the span is measured against
    :func:`time.time` rather than :func:`time.monotonic`: the record is written
    by one process and read by another, and a monotonic reading is a duration
    since this machine booted, which the reading process cannot relate to the
    writer's. That is also why a stamp cannot simply be trusted - the clock it
    was taken from is the current opinion about the date, and an NTP correction
    or a ``date -s`` between the write and the read moves the span by the step.
    A step large enough to put the stamp in this clock's future is reported as
    such rather than as a negative duration, which is what the subtraction
    renders on its own.

    The values refused are the ones a JSON file produces and the subtraction
    does not survive:

    * No ``start_time`` at all, and the ``null`` that says the same thing. Both
      default to ``0`` in the arithmetic, and ``time.time() - 0`` is the epoch:
      a session started minutes ago reads as one running since 1970.
    * ``bool``, which is an ``int`` subclass, so ``true`` is one second past the
      epoch and lands within a second of the same answer.
    * A ``str``, a ``list`` or a ``dict``. ``time.time() - "1788000000"`` raises
      ``TypeError``, and ``float()`` of that same string does not, so the two
      verbs of one tool reach opposite conclusions about one record.
    * ``NaN`` and the infinities, which ``json.load`` produces from a well-formed
      file and which render as ``nan min`` and ``-inf min``.
    * Any stamp at or before the epoch, because ``0`` is the value the absent
      case used to default to and accepting it re-creates that report.

    What is *not* refused is a positive stamp that is merely implausible: ``1``
    is one second past the epoch and is measured as the span it states. The
    boundary is deliberate, because the only non-arbitrary floor available is
    this machine's boot time, and that is not a safe one to compare against - on
    Linux it comes from the ``btime`` recorded in ``/proc/stat`` at boot, which a
    later correction to the clock does not revise, so a backwards correction
    leaves a genuine post-boot stamp reading as earlier than boot. This owner
    grades what a value *is*, not whether a plausible clock produced it.

    Args:
        info: A session record, read for its ``start_time`` key.

    Returns:
        The span in seconds and the text a report's ``Uptime`` field carries. The
        span is ``None`` whenever the record does not state a usable start, and
        the text then says which way it did not, so a caller passes both on
        without deciding anything: the number for a machine, the sentence for
        whoever has to go and look at the file.
    """
    recorded = info.get("start_time")
    if recorded is None:
        return None, "unknown (this record states no start time)"
    if isinstance(recorded, bool) or not isinstance(recorded, int | float):
        return None, f"unknown (this record states a {type(recorded).__name__} as its start time)"
    if not math.isfinite(recorded) or recorded <= 0:
        return None, "unknown (this record's start time is not a wall-clock stamp)"
    span = time.time() - float(recorded)
    if span < 0:
        return None, f"unknown (this record's start time is {-span / 60:.1f} min ahead of this clock)"
    return span, f"{span / 60:.1f} min"


def unusable_pid_result(session_name: str, recorded: Any) -> dict[str, Any]:
    """Refuse a ``stop`` for a session record that names no process id.

    Shared by both session tools so one unusable record reads the same whichever
    one holds it, and split on whether the record carries the key at all: an
    absent pid is the store's own "this record was never given one", while a
    present value that is not a pid is a damaged or hand-edited record, and an
    operator who cannot tell those apart cannot tell which file to look at.

    The type is reported rather than the value, because the type is the mistake:
    a ``float`` is a truncated pid, a ``str`` is a store written by something
    other than these tools, and a ``bool`` is pid 1. The value itself is in the
    file this message points the operator at.

    Args:
        session_name: Name the session is tracked under.
        recorded: The raw value the record carried under ``pid``, or ``None``
            when the key was absent.

    Returns:
        A tool error result whose ``json`` block carries ``pid_usable: False``.
    """
    if recorded is None:
        text = f"No PID found for session '{session_name}'"
    else:
        text = (
            f"Session '{session_name}' records a {type(recorded).__name__} as its PID, and a process id is "
            f"a positive integer, so there is nothing this verb can signal. Recover the process from the "
            f"session store and stop it with 'kill'."
        )
    return {
        "status": "error",
        "content": [
            {"text": text},
            {"json": {"session_name": session_name, "pid_usable": False, "stopped": False}},
        ],
    }


def session_log_path(session_name: str) -> Path:
    """Where a detached session's captured output lives.

    Resolved on each call rather than bound at import, so :data:`SESSION_DIR`
    has one redirect seam covering both the store and the logs beside it.

    Args:
        session_name: The session key the record is stored under.

    Returns:
        The log file path for that session, inside :data:`SESSION_DIR`.
    """
    return SESSION_DIR / f"{session_name}.log"


def store_sessions(sessions_file: Path, sessions: Mapping[str, Any]) -> None:
    """Replace a session store whole, or leave the stored one untouched.

    Both session verbs persist a whole document: :meth:`add_session` and
    :meth:`remove_session` load every record, change one entry, and store the
    map back. So a write that lands partially does not lose the session being
    changed - it loses every session the file held, and the load path reports an
    unparseable store as *no sessions*. Both stores go to the same file, and the
    records in it are the only place a detached process's pid is written down, so
    the loss is not cosmetic: the processes keep running, ``list`` reports none,
    and ``stop`` answers that the session was not found.

    The document is therefore serialized in full *before* the destination is
    touched, and the text is committed through a temp file in the same directory
    plus :func:`os.replace` - the sequence
    :func:`strands_robots.registry.user_registry._save_user_registry` documents
    for its own whole-document store, for the same two reasons:

    * Serializing first is what keeps a rejected write harmless. ``json.dump``
      encodes straight into the stream it is given, so a value it cannot encode
      raises only after a prefix of the new document has replaced the stored one.
    * :func:`os.replace` is atomic within a directory, so a full disk or an I/O
      error during the commit leaves the previous store intact rather than
      truncated, and a concurrent reader observes one whole document or the
      other, never a prefix.

    The temp file is written with :meth:`pathlib.Path.write_text` rather than
    :func:`tempfile.mkstemp` so the store keeps the ordinary umask-derived mode a
    plain ``open(path, "w")`` gave it: replacing its contents is not the moment
    to decide who may read it.

    Args:
        sessions_file: The store to replace. Its parent directory must exist -
            both callers create it when their module loads.
        sessions: The whole session map to store.

    Raises:
        ValueError: A record holds a value JSON cannot represent. Raised before
            the stored store is touched, so it still holds what it last held;
            the originating ``TypeError`` stays on ``__cause__``, naming the
            offending type.
        OSError: The temp file could not be written or renamed. The stored store
            is likewise unchanged, and no temp file is left behind.
    """
    try:
        # Name the encoding the reader names, so the store's spelling is a
        # property of the file rather than of the locale that wrote it.
        payload = json.dumps(dict(sessions), indent=2)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"a session record is not JSON-serializable, so the store cannot be written to "
            f"{sessions_file}: {exc}. The stored sessions are unchanged, so every session "
            f"already recorded stays stoppable. Pass only JSON types (str, int, float, bool, "
            f"None, list, dict)."
        ) from exc

    tmp = sessions_file.with_suffix(sessions_file.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, sessions_file)
    except OSError:
        # The commit did not happen, so the stored document is the previous one.
        # Leaving the temp file behind would put a second, partial store next to
        # it under a name the load path does not read.
        tmp.unlink(missing_ok=True)
        raise


def session_is_running(info: Mapping[str, Any]) -> bool:
    """Whether the process a session record names is still that process.

    Args:
        info: A session record, read through :func:`recorded_pid` for its ``pid``
            and for the identity :data:`PID_STARTED_SINCE_BOOT` names.

    Returns:
        ``True`` while the recorded pid exists *and* still holds the process the
        record was written for. A record naming no usable pid is not running:
        there is no process to ask about, and answering from a converted number
        would answer about a different one.

        A record carrying no identity - one written before it was recorded - can
        only be answered by existence. Of the two ways the identity read can fail,
        only one is an answer: :class:`psutil.NoSuchProcess` means the process went
        away between the two probes, which is the same finished run as a pid that
        was already gone, while :class:`psutil.AccessDenied` means this user may
        not look - and being unable to look is no evidence that the session ended.
    """
    pid = recorded_pid(info)
    if pid is None or not psutil.pid_exists(pid):
        return False
    recorded = info.get(PID_STARTED_SINCE_BOOT)
    if isinstance(recorded, bool) or not isinstance(recorded, int | float):
        return True
    try:
        started = _started_since_boot(pid)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    return abs(started - float(recorded)) <= _IDENTITY_TOLERANCE_S


def reused_pid_result(session_name: str, pid: int) -> dict[str, Any]:
    """Report a ``stop`` for a session whose pid now belongs to another process.

    The session is over - its process exited, which is what let the pid be handed
    out again - so this is the same success as stopping one that had already
    finished. What it must not do is signal the pid, and it says so, because the
    caller cannot otherwise tell this outcome from a kill it asked for.

    Args:
        session_name: Name the session was tracked under.
        pid: The recorded pid, now held by an unrelated process.

    Returns:
        A tool success result whose ``json`` block carries ``pid_reused``.
    """
    return {
        "status": "success",
        "content": [
            {
                "text": f"Session '{session_name}' was already stopped: PID {pid} now belongs to a "
                f"different process, which was not signalled. Its record has been dropped."
            },
            {"json": {"session_name": session_name, "pid": pid, "stopped": True, "pid_reused": True}},
        ],
    }


def confirm_exit(proc: psutil.Process, timeout: float) -> bool | None:
    """Wait up to ``timeout`` seconds for ``proc`` to leave the process table.

    Args:
        proc: The process being stopped. It must have been constructed *before*
            the signal was sent: psutil records the creation time at
            construction, so every probe here is identity-checked and a PID that
            was recycled in the meantime reads as exited rather than as the
            original process still running.
        timeout: Seconds to wait. Returns as soon as the process is gone.

    Returns:
        ``True`` when the process is known to have exited, ``False`` when it is
        known to be still present, and ``None`` when neither could be
        established - this user may not inspect the process
        (:class:`psutil.AccessDenied`). ``None`` must not be read as either
        answer: the PID existing already said it is not death, and being unable
        to look is no evidence that it left.
    """
    try:
        proc.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        return False
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return None
    return True


def unstopped_result(session_name: str, pid: int, verdict: bool | None, doing: str) -> dict[str, Any]:
    """Report a session whose process was not confirmed gone after SIGKILL.

    The caller keeps the session record when it gets this: that store is the
    only place a detached session's PID is written down, so dropping it would
    leave the process running with no supported way left to stop it.

    Args:
        session_name: Name the session is tracked under.
        pid: PID that was signalled.
        verdict: The non-affirmative :func:`confirm_exit` answer being reported -
            ``False`` (still present) or ``None`` (could not be determined).
        doing: What the process is still doing, in the caller's terms
            (``"driving the robot"``, ``"training"``).

    Returns:
        A tool error result whose ``json`` block carries ``stopped`` verbatim, so
        an unknown outcome stays unknown to whoever reads it.
    """
    if verdict is None:
        what = (
            f"Session '{session_name}' (PID {pid}) was signalled with SIGTERM and SIGKILL, but whether it "
            f"exited could not be determined: this user may not inspect it. A session started under sudo "
            f"for device access and stopped as the invoking user reads this way."
        )
    else:
        what = (
            f"Session '{session_name}' (PID {pid}) is still present after SIGTERM and SIGKILL, so it is "
            f"still {doing}. A process that outlives SIGKILL is in an uninterruptible wait."
        )
    return {
        "status": "error",
        "content": [
            {"text": f"{what} Its record is kept so the session stays stoppable; inspect it with 'ps -p {pid}'."},
            {"json": {"session_name": session_name, "pid": pid, "stopped": verdict}},
        ],
    }


class SessionManager:
    """The one reader and writer of the detached-session store.

    Records are keyed by session name and held as one JSON document under
    :data:`SESSION_DIR`. Reading never deletes: :meth:`remove_session` is the
    only thing that drops a record, because this store is the only place a
    detached process's pid is written down, and every read path here is
    load-modify-write - a record a read leaves out is erased from disk by the
    next session started or stopped, after which the process it named goes on
    driving an arm or holding a GPU with no supported way left to stop it.

    Retention costs no accuracy, because presence here is not the running claim.
    ``list`` and ``status`` each derive that from :func:`session_is_running` at
    the moment they are asked, so a retained record reads as running only while
    its pid still holds the process the record was written for.
    """

    def __init__(self) -> None:
        self.sessions_file = SESSION_DIR / "active_sessions.json"

    def _load_sessions(self) -> dict[str, Any]:
        """Every stored record, keyed by session name; no record is dropped.

        Returns:
            The stored records. A store that cannot be read degrades to empty
            rather than raising - including one carrying bytes this encoding
            does not describe, which are read as U+FFFD so a record damaged
            outside its pid still names the process it named (a pid is ASCII).
            The decode policy cannot raise for that reason: the handler below
            answers the two expected failures - the file is gone, or it is not
            JSON - and ``UnicodeDecodeError`` is a ``ValueError`` that passes
            both clauses and would abort the tool action that asked.
        """
        if not self.sessions_file.exists():
            return {}
        try:
            with open(self.sessions_file, encoding="utf-8", errors="replace") as f:
                sessions: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sessions: {e}")
            return {}

        self._report_uninspectable(sessions)
        return sessions

    def _report_uninspectable(self, sessions: Mapping[str, Any]) -> None:
        """Warn for each record this store holds but cannot inspect.

        Two records read that way, and both are kept: one names a pid that
        exists and may not be read, the other names no pid at all. Neither is
        evidence the run ended, and the warning is the operator's only clue that
        ``status`` will report on a process it could not identify.

        ``psutil.AccessDenied`` is the first - a session started under ``sudo``
        for device access and later listed as the invoking user reads this way.
        ``NoSuchProcess`` needs no report: it means the run was reaped between
        the two probes, which is the same finished run as a pid that was already
        gone, and those are retained for their log tail.

        Args:
            sessions: The loaded records. Inspected only; never modified.
        """
        for name, info in sessions.items():
            pid = recorded_pid(info)
            if pid is None:
                if info.get("pid") is not None:
                    # A pid field that is not a process id. Nothing can inspect it,
                    # and converting it would inspect a different process, so the
                    # record is kept and the run reported as not running.
                    logger.warning(
                        "Session '%s' records a %s as its PID, which is not a process id; "
                        "its record is kept, but the run can only be stopped by hand",
                        name,
                        type(info.get("pid")).__name__,
                    )
                continue
            if not psutil.pid_exists(pid):
                continue
            try:
                # Called for what it raises, not for what it returns: the
                # running/finished line is re-derived by ``list`` and ``status``,
                # so this probe exists only to surface a denial.
                psutil.Process(pid).is_running()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                logger.warning(
                    "Session '%s' (PID %s) exists but cannot be inspected; "
                    "keeping its record so the session stays stoppable",
                    name,
                    pid,
                )

    def _save_sessions(self, sessions: Mapping[str, Any]) -> None:
        """Store the record map in full, or leave the stored one untouched.

        :func:`store_sessions` owns the sequence, because losing this store is
        what makes a live session unstoppable.
        """
        try:
            store_sessions(self.sessions_file, sessions)
        except OSError as e:
            logger.error(f"Error saving sessions: {e}")

    def add_session(self, name: str, info: dict[str, Any]) -> None:
        """Persist ``info`` under ``name``, overwriting any record of that name.

        Args:
            name: Session key to store the record under.
            info: Metadata to persist - typically ``pid``,
                :data:`PID_STARTED_SINCE_BOOT`, ``log_file`` and ``start_time``.
        """
        sessions = self._load_sessions()
        sessions[name] = info
        self._save_sessions(sessions)

    def remove_session(self, name: str) -> None:
        """Delete the record stored under ``name``; a no-op when none is.

        Args:
            name: Session key to remove.
        """
        sessions = self._load_sessions()
        if name in sessions:
            del sessions[name]
            self._save_sessions(sessions)

    def get_session(self, name: str) -> dict[str, Any] | None:
        """Return the record stored under ``name``, or ``None`` if none is.

        Args:
            name: Session key to look up.

        Returns:
            The record, whether its process is running or finished; the caller
            derives which from the pid through :func:`session_is_running`.
        """
        return self._load_sessions().get(name)

    def list_sessions(self) -> dict[str, Any]:
        """Return every stored record, keyed by session name.

        Returns:
            Every record the store holds, including those whose process has
            finished - retained so ``status`` can still report the final log
            tail. Being listed is not a claim of running.
        """
        return self._load_sessions()
