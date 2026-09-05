"""A session's running verdict must name the process the record was written for.

``lerobot_train`` and ``lerobot_teleoperate`` run their work detached and write
the pid to an on-disk store that outlives the run that made it. A pid is not a
name: the kernel hands the number back out once the process holding it exits, so
a record that outlives its run points at whatever holds the number next. Both
tools answered "is it running" from ``psutil.pid_exists`` alone, which is a
question about the *number*, so a reused pid read as the session still running -
and ``stop`` acted on that verdict by sending SIGTERM and then SIGKILL to a
process the session never started.

The reuse guard the tree believed it had was ``psutil.Process(pid).is_running()``.
It cannot be one when the object is constructed to ask the question: psutil
records the creation time at construction, so the object carries whatever the pid
means now and agrees with it. ``_process_stop.confirm_exit`` already states the
rule that closes this - the identity has to be captured *before* the question is
asked - and these records captured none, so there was nothing to compare against.

What is recorded now is the process's start offset since boot, which is a
duration and therefore the same number in the run that wrote it and the run that
reads it back hours later. A creation *date* would not be: it is the process's
start ticks plus ``/proc/stat``'s btime, so a correction between the write and the
read would make a live session read as a stranger - the one outcome worse than
this defect, because it would refuse to stop a training run that is holding the
GPU.

The offset is read from the kernel directly - ``/proc/<pid>/stat`` field 22 over
``SC_CLK_TCK`` - and *not* as ``create_time() - boot_time()``. That subtraction
looks like it recovers the ticks but puts the wall clock on both sides, and the
two terms are not guaranteed to be one read of it, so it moves under exactly the
step this identity exists to survive. The last two cells grade that: one pins
equality with the kernel's own value rather than a tolerance any spelling meets,
and one injects the skew and asserts a live session still reads as running.

A record carrying no identity, or one whose identity cannot be read, still falls
back to existence. Those are pinned below too: they are the reason this change
cannot strand a session, and they are the limit of the fix.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

import psutil
import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools._process_stop import (
    _IDENTITY_TOLERANCE_S,
    PID_STARTED_SINCE_BOOT,
    process_started_since_boot,
    session_is_running,
)

#: The two session tools, the extra keywords each needs to be called at all, and
#: the session name and ``action`` value a record of theirs carries.
TOOLS = [
    pytest.param(train_mod, {"dataset_root": "/unused"}, "training", "train", id="train"),
    pytest.param(tele_mod, {}, "arm_teleop", "teleoperate", id="teleoperate"),
]

#: How far a stale record's identity sits from the live process holding its pid.
#: Any nonzero gap is a different process; an hour is the scale a real one differs
#: by, since the reused number was handed out after the recorded process exited.
STALE_GAP_S = 3600.0


def _tool(module: Any) -> Any:
    """The tool callable of a session module."""
    return getattr(module, module.__name__.rsplit(".", 1)[-1])


@pytest.fixture(autouse=True)
def _isolate_session_dirs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both stores at a temp dir so no test touches a real session."""
    for module in (train_mod, tele_mod):
        session_dir = tmp_path / module.__name__.rsplit(".", 1)[-1]
        session_dir.mkdir()
        monkeypatch.setattr(module, "SESSION_DIR", session_dir)


#: The real ``os.kill``, bound before any test can replace it. A cell that pins
#: which signals ``stop`` sends replaces ``os.kill`` for the module under test,
#: which is this same module object, and that patch is not necessarily undone
#: before the fixture below cleans up - so the cleanup must not go through the
#: name being patched, or it waits out the process's whole sleep.
_REAL_KILL = os.kill


@pytest.fixture
def stranger() -> Any:
    """A live process no session started - what a reused pid points at.

    Detached with its own session so a signal aimed at it is not also aimed at
    this test's process group, and reaped here rather than by whatever probes it.
    """
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    deadline = time.monotonic() + 5.0
    while process_started_since_boot(proc.pid) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert psutil.pid_exists(proc.pid), "premise: the stranger must be running"
    yield proc
    if psutil.pid_exists(proc.pid):
        _REAL_KILL(proc.pid, signal.SIGKILL)
    proc.wait()


def _record(pid: int, action: str, identity: float | None | object = None) -> dict[str, Any]:
    """A session record for ``pid``, optionally carrying a recorded identity."""
    info: dict[str, Any] = {
        "action": action,
        "pid": pid,
        "start_time": time.time() - STALE_GAP_S,
        "policy_type": "act",
        "output_dir": "/tmp/out",
        "robot_type": "so101_follower",
        "teleop_type": "so101_leader",
    }
    if identity is not None:
        info[PID_STARTED_SINCE_BOOT] = identity
    return info


def _seed(module: Any, name: str, info: dict[str, Any]) -> Any:
    """A store of ``module`` holding one record, and the manager over it."""
    mgr = module.SessionManager()
    mgr.add_session(name, info)
    assert name in json.loads(mgr.sessions_file.read_text()), "premise: the record must reach disk"
    return mgr


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in result["content"] if "text" in block)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


# ---------------------------------------------------------------------------
# The identity itself.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc for the kernel's own value")
def test_the_identity_is_the_tick_count_the_kernel_recorded(stranger: Any) -> None:
    """Why a correction to the wall clock cannot move it: it is not a date.

    Field 22 of ``/proc/<pid>/stat`` is the process's start in ticks since boot,
    which the kernel writes once and never revises. Recovering exactly that from
    psutil is what makes the recorded identity comparable across the hours - and
    across a clock step - between the run that writes it and the run that reads it.
    """
    with open(f"/proc/{stranger.pid}/stat", "rb") as handle:
        after_comm = handle.read().rpartition(b")")[2].split()
    ticks = int(after_comm[19])

    started = process_started_since_boot(stranger.pid)

    assert started is not None, "a live process this user owns must be identifiable"
    assert abs(started - ticks / os.sysconf("SC_CLK_TCK")) < 0.01, (
        f"the identity must be the kernel's own start ticks, got {started}"
    )


def test_two_live_processes_do_not_share_an_identity(stranger: Any) -> None:
    """It has to distinguish processes, and it has to be stable while one lives."""
    assert process_started_since_boot(stranger.pid) != process_started_since_boot(os.getpid())
    assert process_started_since_boot(stranger.pid) == process_started_since_boot(stranger.pid)


@pytest.mark.parametrize(
    ("identity", "expected", "why"),
    [
        pytest.param("live", True, "the record names this process", id="matches"),
        pytest.param("stale", False, "the pid was reused; the session's process is gone", id="reused"),
        pytest.param(None, True, "a record from before the identity was written down", id="absent"),
        pytest.param("denied", True, "an identity that cannot be read is not a mismatch", id="uninspectable"),
    ],
)
def test_the_running_verdict_reads_the_identity_and_not_only_the_number(
    stranger: Any, monkeypatch: pytest.MonkeyPatch, identity: str | None, expected: bool, why: str
) -> None:
    """The whole verdict, over every shape a record's identity can take."""
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    recorded: float | None = {"live": live, "stale": live - STALE_GAP_S, "denied": live, None: None}[identity]
    if identity == "denied":
        monkeypatch.setattr(psutil, "Process", _denied_process(psutil.AccessDenied))

    assert session_is_running(_record(stranger.pid, "train", recorded)) is expected, why


def test_a_pid_that_does_not_exist_is_not_running() -> None:
    """The half that already worked, kept as a control."""
    gone = _a_dead_pid()
    assert session_is_running(_record(gone, "train", 1.0)) is False


def _a_dead_pid() -> int:
    """A pid that has certainly exited: one this test spawned and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])  # noqa: S603 - fixed argv, no shell
    proc.wait()
    return proc.pid


def _denied_process(exc: type[Exception]) -> Any:
    """A ``psutil.Process`` stand-in that refuses every probe of the process.

    It refuses ``create_time`` as well as ``is_running`` and ``wait``: a stand-in
    that answered one and not the others would model a process no kernel produces,
    and the identity check reads the first of them.
    """

    class _Denied:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def create_time(self) -> float:
            raise exc(self._pid)

        def is_running(self) -> bool:
            raise exc(self._pid)

        def wait(self, timeout: float | None = None) -> int:
            raise exc(self._pid)

    return _Denied


# ---------------------------------------------------------------------------
# What the two tools report.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("module", "extra", "name", "action"), TOOLS)
@pytest.mark.parametrize("verb", ["list", "status"])
@pytest.mark.parametrize(
    ("gap", "running"),
    [pytest.param(0.0, True, id="own-process"), pytest.param(STALE_GAP_S, False, id="reused-pid")],
)
def test_a_reused_pid_is_not_reported_as_a_running_session(
    stranger: Any, module: Any, extra: dict[str, Any], name: str, action: str, verb: str, gap: float, running: bool
) -> None:
    """Both tools, both read verbs: the running claim follows the identity.

    The two stores dispose of the finished record differently - the training store
    keeps it for its log tail and the teleoperation store prunes it - and that
    difference is theirs to keep. What neither may do is call it running.
    """
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    _seed(module, name, _record(stranger.pid, action, live - gap))

    result = _tool(module)(action=verb, session_name=name, **extra)

    assert ("Status: Running" in _text(result)) is running, _text(result)


@pytest.mark.parametrize(("module", "extra", "name", "action"), TOOLS)
def test_stop_does_not_signal_a_process_the_record_no_longer_names(
    stranger: Any, module: Any, extra: dict[str, Any], name: str, action: str
) -> None:
    """The consequence a report cannot show: the signals reach a stranger.

    Nothing is stubbed here - the pid is a real process this test started, and the
    assertion is that it is still there afterwards. That is what a reused pid is:
    a live process the session did not start.
    """
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    mgr = _seed(module, name, _record(stranger.pid, action, live - STALE_GAP_S))

    _tool(module)(action="stop", session_name=name, **extra)

    assert psutil.pid_exists(stranger.pid), "stop killed a process the session never started"
    assert stranger.poll() is None, "the stranger must not have been signalled"
    assert mgr.get_session(name) is None, "the finished session's record is dropped"


def test_stop_says_it_did_not_signal_the_pid_it_was_given(stranger: Any) -> None:
    """The report the caller needs to tell this outcome from a kill it asked for.

    Read on the training store, which is where the refusal is reachable: it keeps
    a finished record, so ``stop`` gets to look at one. The teleoperation store
    prunes it first, and the cell below pins that instead.
    """
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    _seed(train_mod, "training", _record(stranger.pid, "train", live - STALE_GAP_S))

    result = train_mod.lerobot_train(action="stop", session_name="training", dataset_root="/unused")

    assert result["status"] == "success", _text(result)
    assert _json(result)["pid_reused"] is True, _text(result)
    assert "was not signalled" in _text(result), _text(result)


@pytest.mark.parametrize(("module", "extra", "name", "action"), TOOLS)
def test_stop_still_signals_the_process_the_record_does_name(
    stranger: Any, monkeypatch: pytest.MonkeyPatch, module: Any, extra: dict[str, Any], name: str, action: str
) -> None:
    """The control: a session whose pid is still its own process is stopped."""
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    _seed(module, name, _record(stranger.pid, action, live))
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    result = _tool(module)(action="stop", session_name=name, **extra)

    sent = [entry for entry in signalled if entry[1] != 0]
    assert sent and sent[0] == (stranger.pid, signal.SIGTERM), f"stop must signal the recorded pid, sent {sent}"
    assert _json(result).get("pid_reused") is None, "this pid was not reused"


# ---------------------------------------------------------------------------
# The teleoperation store's prune, which exists to drop exactly this record.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("gap", "survives"),
    [pytest.param(0.0, True, id="own-process"), pytest.param(STALE_GAP_S, False, id="reused-pid")],
)
def test_the_teleop_prune_drops_a_record_whose_pid_was_reused(stranger: Any, gap: float, survives: bool) -> None:
    """A prune that keeps a record because the number exists prunes nothing."""
    live = process_started_since_boot(stranger.pid)
    assert live is not None, "premise"
    mgr = _seed(tele_mod, "arm_teleop", _record(stranger.pid, "teleoperate", live - gap))

    assert ("arm_teleop" in mgr.list_sessions()) is survives


# ---------------------------------------------------------------------------
# Recording it in the first place.
# ---------------------------------------------------------------------------
def test_start_records_the_identity_of_the_process_it_started(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stranger: Any
) -> None:
    """Without this the comparison above has nothing to compare against.

    The training process is stood in for by the stranger fixture - a real process,
    so the identity written down is a real one - and the assertion is that the
    record carries that process's own identity rather than nothing.
    """
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 8}))

    class _Started:
        pid = stranger.pid

    monkeypatch.setattr(train_mod.subprocess, "Popen", lambda cmd, **kwargs: _Started())

    result = train_mod.lerobot_train(
        action="start",
        dataset_root=str(root),
        policy_type="act",
        output_dir=str(tmp_path / "out"),
        session_name="training",
    )

    assert result["status"] == "success", _text(result)
    stored = train_mod.SessionManager().get_session("training")
    assert stored is not None
    assert stored[PID_STARTED_SINCE_BOOT] == process_started_since_boot(stranger.pid)


def _kernel_ticks_since_boot(pid: int) -> float:
    """Field 22 of ``/proc/<pid>/stat`` in seconds - the kernel's own record.

    Everything up to and including ``comm`` is dropped by partitioning on the
    last ``)``, because a process name may itself contain spaces and parentheses.
    """
    with open(f"/proc/{pid}/stat", "rb") as handle:
        after_comm = handle.read().rpartition(b")")[2].split()
    return float(after_comm[19]) / os.sysconf("SC_CLK_TCK")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs identity is Linux-only")
def test_the_identity_is_exactly_the_kernel_value_not_merely_close(stranger: Any) -> None:
    """Equality, not a tolerance: the read must not go through the wall clock.

    ``test_the_identity_is_the_tick_count_the_kernel_recorded`` above allows
    0.01 s, which any ``create_time() - boot_time()`` spelling also satisfies on
    an undisturbed clock - so that cell cannot tell the two implementations
    apart. Reading field 22 directly returns the kernel's own value bit for bit,
    and pinning equality is what refuses a reintroduced subtraction.
    """
    assert process_started_since_boot(stranger.pid) == _kernel_ticks_since_boot(stranger.pid)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs identity is Linux-only")
def test_the_identity_survives_a_step_of_the_wall_clock(stranger: Any, monkeypatch: Any) -> None:
    """The graded property: an NTP correction must not move this number.

    The identity exists to be comparable between the run that writes it and a run
    that reads it back hours later, so surviving a clock step is the whole point
    (AGENTS.md > Review Learnings (#86) > "Clocks: a duration is measured, a
    stamp is recorded"). ``create_time() - boot_time()`` does not have that
    property, because the two terms are not guaranteed to be the same read of
    btime: on psutil at or before 7.0 the process side adds a btime cached at
    import while top-level ``boot_time()`` deliberately re-reads ``/proc/stat``,
    and on 7.2 both re-read it, leaving a step that lands between the two reads.

    The skew is injected rather than waited for: ``psutil.boot_time`` is moved so
    the two terms of that subtraction disagree by 10 s, which is what a
    correction between the cache and the read produces. Measured against the
    pre-fix spelling on this tree, the value moved by exactly the step size and a
    live session mismatched its own record - `stop` would then take the
    ``pid_reused`` branch and drop the record, and the teleop prune would write
    it out of the store, leaving the process running with no supported way to
    stop it. Reading field 22 has no wall-clock term to disagree about.
    """
    truth = _kernel_ticks_since_boot(stranger.pid)
    record = {"pid": stranger.pid, PID_STARTED_SINCE_BOOT: truth}
    assert session_is_running(record) is True, "control: the stranger is alive before the step"

    real_boot_time = psutil.boot_time
    step_s = 10.0
    monkeypatch.setattr(psutil, "boot_time", lambda: real_boot_time() - step_s)

    # The spelling this test refuses, evaluated here so the cell carries its own
    # evidence that the step is large enough to matter rather than asserting it.
    skewed = psutil.Process(stranger.pid).create_time() - psutil.boot_time()
    assert abs(skewed - truth) > _IDENTITY_TOLERANCE_S, (
        f"the injected step must exceed the tolerance to grade anything, moved {skewed - truth}"
    )

    assert process_started_since_boot(stranger.pid) == truth, "a clock step moved the identity"
    assert session_is_running(record) is True, "a clock step made a live session read as a stranger"
