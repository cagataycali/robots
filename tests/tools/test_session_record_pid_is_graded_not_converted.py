"""A session record's pid is graded, not converted into a nearby process id.

Both session tools keep their sessions in a JSON file, and every verb that
reports on or stops one starts by reading a ``pid`` out of it. That value is a
file's declaration rather than a caller's argument, and converting it is how a
record comes to name a process it was never written for: ``int(4321.5)`` is
``4321``, and ``bool`` is an ``int`` subclass, so ``true`` is pid 1 - ``init`` on
Linux. Measured before the fix these tests pin, a store whose ``pid`` field held
``true`` read as a live session and ``stop`` handed ``os.kill`` pid 1, while a
record spelling a live stranger's pid with ``.5`` after it got that stranger
killed and reported success naming a number that is not a pid.

The same conversion is the other way a record goes wrong. ``int()`` raises for
values ``json.load`` produces from a well-formed file - ``ValueError`` for a
string carrying the U+FFFD the store's own decode policy substitutes for a
damaged byte, ``OverflowError`` for ``1e400``, ``ValueError`` for ``NaN`` - and
neither store's read handles either, so a read documented to degrade to "no
sessions" aborted the action that asked instead.

:func:`~strands_robots.tools._process_stop.recorded_pid` is the one reader of
that field for both tools, in the shape
:func:`~strands_robots.utils.declared_count` uses for a count a file declares:
the pid, or ``None`` for no usable one. These tests pin the domain, that no
spelling raises anywhere, that nothing outside it is ever signalled, and that
the one spelling a writer produces still stops its session.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools import _process_stop
from strands_robots.tools._process_stop import session_is_running

#: A pid this process certainly holds, so "exists" is settled and the only thing
#: under test is the spelling it is written down in.
LIVE_PID = os.getpid()

#: Spellings of a pid a store can carry, with the reason each is refused. A list
#: of pairs rather than a dict because ``0 == False`` and ``4321 == 4321.0``
#: collapse in a mapping exactly the distinctions a conversion defect is about.
UNUSABLE: list[tuple[str, Any, str]] = [
    ("true", True, "bool is an int subclass, so int() of it is pid 1 - init on Linux"),
    ("half above a live pid", float(LIVE_PID) + 0.5, "int() truncates to a pid the record does not carry"),
    ("str digits", str(LIVE_PID), "psutil.pid_exists raises TypeError on a str, so int() hid a store it refuses"),
    ("padded str digits", f" {LIVE_PID} ", "int() strips whitespace, so two spellings of one store are accepted"),
    ("str with a replaced byte", "12\ufffd", "what the store's errors='replace' decode leaves in a damaged pid"),
    ("inf", float("inf"), "1e400 is a well-formed JSON number; int() of it raises OverflowError"),
    ("nan", float("nan"), "int() of it raises ValueError"),
    ("zero", 0, "os.kill reads 0 as every process in the caller's own process group"),
    ("negative", -LIVE_PID, "os.kill reads a negative number as a process group"),
    ("above pid_t", 2**31, "psutil.pid_exists and os.kill raise OverflowError above the signed 32-bit ceiling"),
    ("a list", [LIVE_PID], "int() of it raises TypeError"),
    ("absent", None, "the record was never given a pid"),
]
UNUSABLE_IDS = [row[0] for row in UNUSABLE]


def _record(pid: Any) -> dict[str, Any]:
    """A session record carrying ``pid``, or none at all when it is ``None``."""
    info: dict[str, Any] = {"action": "teleoperate", "start_time": 0.0}
    if pid is not None:
        info["pid"] = pid
    return info


@pytest.fixture(autouse=True)
def _isolate_both_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both tools' session stores at a temp dir, never the tree."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)
    return session_dir


def _write_store(session_dir: Path, name: str, pid: Any) -> None:
    """Write a one-record store, the way a hand-edited or damaged file reads."""
    (session_dir / "active_sessions.json").write_text(json.dumps({name: _record(pid)}))


class _RecordingOs:
    """The real ``os`` module with ``kill`` recorded instead of sent.

    ``monkeypatch.setattr(os, "kill", ...)`` is not usable here: psutil answers
    ``pid_exists`` on POSIX by asking ``os.kill(pid, 0)``, so the recorder would
    also be answering the existence probe these verbs take first. Replacing the
    tool module's own ``os`` name leaves psutil's reference alone.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self.sent: list[tuple[Any, int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def kill(self, pid: Any, sig: int) -> None:
        self.sent.append((pid, int(sig)))
        raise ProcessLookupError(3, "recorded, not sent")


@pytest.fixture
def recorded_signals(monkeypatch: pytest.MonkeyPatch) -> _RecordingOs:
    """Record what the teleop stop verb hands ``os.kill``, without sending it."""
    proxy = _RecordingOs(os)
    monkeypatch.setattr(tele_mod, "os", proxy)
    return proxy


# --------------------------------------------------------------------------- #
# the domain                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("_label", "pid", "_why"), UNUSABLE, ids=UNUSABLE_IDS)
def test_a_spelling_outside_the_domain_names_no_pid(_label: str, pid: Any, _why: str) -> None:
    """Every unusable spelling answers "no pid", never a nearby number."""
    assert _process_stop.recorded_pid(_record(pid)) is None


def test_the_spelling_a_writer_produces_is_the_pid() -> None:
    """An integer pid passes through unchanged - the domain narrows nothing real.

    Both tools write ``proc.pid`` into the store, so this is the only spelling a
    record can be born with, and it must survive the grading intact.
    """
    assert _process_stop.recorded_pid(_record(LIVE_PID)) == LIVE_PID


# --------------------------------------------------------------------------- #
# no spelling raises, and none of them reads as running                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("_label", "pid", "_why"), UNUSABLE, ids=UNUSABLE_IDS)
def test_an_unusable_pid_is_not_running_rather_than_an_exception(_label: str, pid: Any, _why: str) -> None:
    """The running question has an answer for every spelling a store can hold.

    Before the fix four of these raised out of ``session_is_running``
    (``ValueError``, ``OverflowError``, ``TypeError``) and three answered
    ``True`` on the strength of a converted number.
    """
    assert session_is_running(_record(pid)) is False


def test_a_live_integer_pid_still_reads_as_running() -> None:
    """The one spelling that names a process is still answered from the process."""
    assert session_is_running(_record(LIVE_PID)) is True


# --------------------------------------------------------------------------- #
# both stores degrade rather than aborting the action that asked              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("_label", "pid", "_why"), UNUSABLE, ids=UNUSABLE_IDS)
def test_the_store_keeps_an_unusable_record_without_raising(
    _isolate_both_stores: Path, _label: str, pid: Any, _why: str
) -> None:
    """A record naming no process is kept, and read the same way by both tools.

    Retention is the point: the record is the only place a detached run's pid is
    written down. Before the fix ``psutil.pid_exists`` was handed the raw value
    and raised ``TypeError`` for a ``str``, a ``float`` and a ``list``. Both
    tools read one store through one manager, so a spelling cannot read as a
    live session in one and as an aborted read in the other.
    """
    _write_store(_isolate_both_stores, "run", pid)

    assert list(tele_mod.SessionManager().list_sessions()) == ["run"]
    assert list(train_mod.SessionManager().list_sessions()) == ["run"]


def test_a_store_whose_pid_field_is_undecodable_still_names_the_record(_isolate_both_stores: Path) -> None:
    """The bytes the decode policy was chosen for reach the pid, and still degrade.

    The store is read with ``errors="replace"`` precisely so a damaged file
    degrades instead of raising, and documents that. The substitute lands in
    whichever field carried the damaged byte, and when that field is the pid,
    ``int()`` of the result raised ``ValueError`` - which the read does not
    handle. The record survives the read either tool takes, because a damaged
    pid is the case where the record is the only thing left naming the process.
    """
    store = _isolate_both_stores / "active_sessions.json"
    store.write_bytes(b'{"arm": {"pid": "12\xff34", "action": "teleoperate", "start_time": 0.0}}')

    assert list(train_mod.SessionManager().list_sessions()) == ["arm"]
    assert list(tele_mod.SessionManager().list_sessions()) == ["arm"]


# --------------------------------------------------------------------------- #
# nothing outside the domain is signalled                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("_label", "pid", "_why"), UNUSABLE, ids=UNUSABLE_IDS)
def test_stop_signals_nothing_for_a_record_that_names_no_pid(
    _isolate_both_stores: Path, recorded_signals: _RecordingOs, _label: str, pid: Any, _why: str
) -> None:
    """No spelling outside the domain reaches a signal.

    ``true`` reached ``os.kill(1, SIGTERM)`` before the fix - measured, with the
    signal recorded rather than sent, which is also why this test records it.
    """
    _write_store(_isolate_both_stores, "arm", pid)

    result = tele_mod.lerobot_teleoperate(action="stop", session_name="arm")

    assert result["status"] == "error"
    assert recorded_signals.sent == [], f"a record spelling its pid as {_label} was signalled"


def test_stop_still_signals_a_session_whose_pid_is_an_integer(
    _isolate_both_stores: Path, recorded_signals: _RecordingOs
) -> None:
    """The control the refusals are measured against: a real pid is still stopped.

    Without this the assertion above passes on a verb that signals nothing at
    all.
    """
    _write_store(_isolate_both_stores, "arm", LIVE_PID)

    tele_mod.lerobot_teleoperate(action="stop", session_name="arm")

    assert recorded_signals.sent == [(LIVE_PID, int(signal.SIGTERM))]


def test_a_live_stranger_is_not_killed_by_a_record_that_spells_its_pid_as_a_float(
    _isolate_both_stores: Path,
) -> None:
    """The harm, with real signals: a process nothing here started stays alive.

    ``int(4321.5)`` is a pid, so ``stop`` used to signal whatever held it, wait
    for it to exit, and report success quoting ``PID: 4321.5`` - a number that is
    not a process id. Pinned against a child of this test rather than a mock,
    because "the process died" is the claim.
    """
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(50):  # let it reach the sleep, so an exit means a signal
            if psutil.pid_exists(stranger.pid):
                break
            time.sleep(0.02)
        _write_store(_isolate_both_stores, "arm", float(stranger.pid) + 0.5)

        tele_mod.lerobot_teleoperate(action="stop", session_name="arm")
        time.sleep(0.3)

        assert stranger.poll() is None, "stop killed a process the session record does not name"
    finally:
        stranger.kill()
        stranger.wait(timeout=5)


# --------------------------------------------------------------------------- #
# what the operator is told                                                   #
# --------------------------------------------------------------------------- #
def test_an_absent_pid_and_an_unusable_one_are_refused_in_their_own_words(_isolate_both_stores: Path) -> None:
    """Two records, two reports: never given a pid, versus given a damaged one.

    The training store keeps both, so both refusals are reachable there. An
    operator who cannot tell them apart cannot tell whether to look for a running
    process at all.
    """
    (_isolate_both_stores / "active_sessions.json").write_text(
        json.dumps({"nopid": _record(None), "bad": _record(4321.5)})
    )

    absent = train_mod.lerobot_train(action="stop", dataset_root="/x", session_name="nopid")
    unusable = train_mod.lerobot_train(action="stop", dataset_root="/x", session_name="bad")

    assert absent["status"] == unusable["status"] == "error"
    assert "No PID found for session 'nopid'" in absent["content"][0]["text"]
    assert "records a float as its PID" in unusable["content"][0]["text"], "the type is the mistake"
    assert unusable["content"][1]["json"] == {"session_name": "bad", "pid_usable": False, "stopped": False}


# --------------------------------------------------------------------------- #
# one reader, and it stays the only one                                       #
# --------------------------------------------------------------------------- #
def test_no_session_tool_converts_a_recorded_pid() -> None:
    """No module reads a stored pid through ``int()``, so the domain cannot be bypassed.

    An AST scan rather than a list of the modules known today: the two session
    tools and their shared stop helper all held ``int(pid)``, and a third tool
    keeping a session store would have to be added to a hand-written roster
    before this could catch it.
    """
    tools_dir = Path(train_mod.__file__).parent
    converts: list[str] = []
    for module in sorted(tools_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(module.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int"):
                continue
            if node.args and "pid" in ast.unparse(node.args[0]).lower():
                converts.append(f"{module.name}:{node.lineno}: {ast.unparse(node)}")

    assert converts == [], "a stored pid is converted rather than graded: " + "; ".join(converts)
