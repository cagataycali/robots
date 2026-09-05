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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import psutil

# How long to let a process wind itself down after SIGTERM before escalating.
# The session processes this covers flush a dataset shard or a checkpoint on
# the way out, so the grace period is real work, not politeness.
SIGTERM_GRACE_S = 2.0

# How long to wait for SIGKILL to take effect before reporting that it has not.
# A process still present after this is in an uninterruptible wait; more waiting
# does not change the verdict, and the caller needs the verdict.
SIGKILL_CONFIRM_S = 2.0

#: Session-record key holding the identity of the process the record was written
#: for: how long after boot that process started.
PID_STARTED_SINCE_BOOT = "pid_started_since_boot"

# Tolerance for comparing two reads of that identity. Both are the same integer
# tick count the kernel recorded, divided by the tick rate, so they differ only by
# the float error of the subtraction below - about 1e-7 s on a month of uptime.
# One tick is far above that noise, and far below the lifetime of any session,
# which is the gap a reused pid's own start offset differs by.
_IDENTITY_TOLERANCE_S = 0.05


def process_started_since_boot(pid: int) -> float | None:
    """How long after boot the process holding ``pid`` started.

    A start offset rather than a creation date, because the two ends of the
    comparison it serves sit in different processes minutes or hours apart: the
    session store is written by the run that started the process and read back by
    a later one. ``create_time()`` is a date - ``/proc/stat``'s ``btime`` plus the
    process's own start ticks, and ``btime`` is recomputed from the current wall
    clock on every read - so an NTP correction or a ``date -s`` between the write
    and the read moves it by the size of the step, and a live session would read
    as a stranger. Subtracting the boot time recovers the tick count itself,
    which is a duration and moves for nothing.

    Args:
        pid: The pid to identify.

    Returns:
        Seconds between boot and the process's creation, or ``None`` when the
        process is gone or this user may not inspect it. ``None`` is not evidence
        either way, and callers must not read it as a mismatch.
    """
    try:
        return psutil.Process(pid).create_time() - psutil.boot_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def session_is_running(info: Mapping[str, Any]) -> bool:
    """Whether the process a session record names is still that process.

    Args:
        info: A session record, read for its ``pid`` and for the identity
            :data:`PID_STARTED_SINCE_BOOT` names.

    Returns:
        ``True`` while the recorded pid exists *and* still holds the process the
        record was written for.

        A record carrying no identity - one written before it was recorded - can
        only be answered by existence. Of the two ways the identity read can fail,
        only one is an answer: :class:`psutil.NoSuchProcess` means the process went
        away between the two probes, which is the same finished run as a pid that
        was already gone, while :class:`psutil.AccessDenied` means this user may
        not look - and being unable to look is no evidence that the session ended.
    """
    pid = info.get("pid")
    if not pid or not psutil.pid_exists(int(pid)):
        return False
    recorded = info.get(PID_STARTED_SINCE_BOOT)
    if isinstance(recorded, bool) or not isinstance(recorded, int | float):
        return True
    try:
        started = psutil.Process(int(pid)).create_time() - psutil.boot_time()
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
