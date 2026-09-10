"""The session store must not delete the record of a live process.

One store holds every detached robot session: ``lerobot_train`` and
``lerobot_teleoperate`` both start their runs with ``start_new_session=True`` and
write the pid into the same JSON document, read and written by the one
:class:`~strands_robots.tools._process_stop.SessionManager`. It is the only place
that pid is recorded, and ``list``, ``status`` and ``stop`` all look the session
up there. So what the load step chooses to leave out is load-bearing, and it is
load-bearing twice over, because ``add_session`` and ``remove_session`` are
load-modify-write - a record the load omits is erased from disk by the next
session started or stopped, after which the process holds the GPU or goes on
driving the arm with no supported way left to stop it.

That is also why the store cannot hold two retention policies. A read that
prunes prunes for every tool reading the same document, so the destructive policy
is the one that takes effect: measured before this rule reached one owner, three
records the training tool documented itself as keeping were erased from disk by
one teleoperation ``list``, and the same process logged "its record is kept" and
"dropping the record" about the same record.

Leaving a record out is not needed to avoid over-reporting it either: presence in
the store is not the running claim. ``list`` and ``status`` each derive that from
:func:`~strands_robots.tools._process_stop.session_is_running` at the moment they
are asked, so a retained record reads as running only while its pid *still holds
the process the record was written for*. Both halves of that are pinned below as
controls: a pid that is gone, and a pid that another process now holds.

The load path takes two probes of its own, and neither is that verdict.
``psutil.pid_exists`` answers existence with a signal; ``Process(pid).is_running()``
reads the process, and ``_report_uninspectable`` calls it for what it raises
rather than for what it returns. No outcome of either is grounds for deleting the
record:

* ``pid_exists`` False - the run finished. Kept, so ``status`` can still report
  the final log tail.
* an identity the pid no longer reports - the run finished and the number was
  handed out again. The same finished run as the row above, reached through the
  identity comparison rather than through existence.
* ``NoSuchProcess`` - reaped between the two probes. Also the same finished run,
  and which of these paths a finished run takes is a race, so they must not be
  classified differently.
* ``AccessDenied`` - the process exists and this user may not inspect it; a
  session started under ``sudo`` for device access and later listed as the
  invoking user reads this way. That is not death, and it is the one case where
  dropping the record loses a pid that still names a *live* process.

No case below poses ``is_running()`` answering ``False``, because that call
cannot: a freshly constructed :class:`psutil.Process` captures the identity it is
about to be asked about, so on the locked psutil it answers ``True`` or raises
:class:`~psutil.NoSuchProcess`, and the load path constructs one per read. Only an
object built before the exit and read after it returns ``False``. The stand-ins
below therefore refuse the probe rather than answering it.

Deletion still happens, on request: ``remove_session`` is the one thing that
drops a record, and a ``stop`` that reaches its process removes it. Retention is
about classification, not about a store that can never shrink.
"""

from __future__ import annotations

import ast
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools import _process_stop

SessionManager = train_mod.SessionManager
lerobot_train = train_mod.lerobot_train

# Required by the tool signature and never read by the session actions. Spelled
# at each call rather than splatted from a dict, which mypy cannot match against
# the tool's typed keywords.
UNUSED_DATASET = "/unused"


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the session store to a temp dir so no test touches the tree."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)
    return session_dir


def _live_pid() -> int:
    """A pid that certainly exists and that we own: this test process."""
    pid = os.getpid()
    assert train_mod.psutil.pid_exists(pid), "premise: the test process must exist"
    return pid


def _free_pid() -> int:
    """A pid no process holds, so the host's own verdict for it is "finished".

    Taken from the top of the kernel's range rather than a fixed low number: a
    low one is exactly what a busy machine is likely to have handed out, which is
    how a test's verdict comes to depend on the host it runs on.
    """
    for candidate in range(4194303, 4194303 - 256, -1):
        if not train_mod.psutil.pid_exists(candidate):
            return candidate
    raise AssertionError("premise: some pid near the top of the range must be free")


def _raise_on_probe(monkeypatch: pytest.MonkeyPatch, exc: type[Exception]) -> None:
    """Make every probe of the process raise ``exc`` while ``pid_exists`` stays truthful.

    ``stop`` confirms the exit with ``Process.wait()``, so a stand-in for an
    uninspectable process has to refuse that the same way it refuses
    ``is_running()``; answering one and not the other would model a process no
    kernel produces.
    """

    class _Probe:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def is_running(self) -> bool:
            raise exc(self._pid)

        def wait(self, timeout: float | None = None) -> int:
            raise exc(self._pid)

    monkeypatch.setattr(train_mod.psutil, "Process", _Probe)


def _stored(mgr: Any) -> dict[str, Any]:
    """The records the store holds on disk, independent of what a load returns."""
    if not mgr.sessions_file.exists():
        return {}
    return json.loads(mgr.sessions_file.read_text())


def _seed(name: str = "training", **extra: Any) -> tuple[Any, int]:
    """A store holding one session whose pid is live, and that pid."""
    mgr = SessionManager()
    pid = _live_pid()
    mgr.add_session(name, {"action": "train", "pid": pid, "start_time": 0.0, **extra})
    assert name in _stored(mgr), "premise: the session must reach disk"
    return mgr, pid


# ---------------------------------------------------------------------------
# A process that exists but cannot be inspected keeps its record.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("read", "why"),
    [
        pytest.param(lambda mgr: "training" in mgr.list_sessions(), "list must show it", id="list"),
        pytest.param(lambda mgr: mgr.get_session("training") is not None, "stop looks it up here", id="get"),
    ],
)
def test_an_uninspectable_live_session_is_still_visible(monkeypatch: pytest.MonkeyPatch, read: Any, why: str) -> None:
    """``AccessDenied`` on a pid that exists must not hide the session."""
    mgr, _ = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    assert read(mgr), f"a session whose pid exists must stay visible when the deeper probe is denied: {why}"


@pytest.mark.parametrize(
    ("mutate", "id_"),
    [
        pytest.param(lambda mgr: mgr.add_session("second", {"pid": 4242, "action": "train"}), "starting", id="start"),
        pytest.param(lambda mgr: mgr.remove_session("second"), "stopping", id="stop"),
    ],
)
def test_a_write_path_does_not_erase_an_uninspectable_sibling(
    monkeypatch: pytest.MonkeyPatch, mutate: Any, id_: str
) -> None:
    """The consequence a read-only check cannot see: the omission reaches disk.

    ``add_session`` and ``remove_session`` both load, modify and write back, so a
    record missing from the load is deleted from the store by the next session
    started or stopped - not by anything the operator did to *this* session.
    """
    mgr, _ = _seed()
    mgr.add_session("second", {"pid": 4242, "action": "train"})
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    mutate(mgr)

    assert "training" in _stored(mgr), (
        f"{id_} another session erased the record of one whose probe was denied, "
        "and that store is the only place its pid was written down"
    )


def test_an_uninspectable_session_can_still_be_deleted_on_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inversion this rule removes, in its sharpest form.

    ``remove_session`` deletes ``name`` only if the load it does first returned
    it, so a record the load omitted could not be deleted *on request* - while
    the very same omission deleted it as a side effect of touching an unrelated
    session. Retention makes the explicit removal the thing that works, which is
    also what keeps retention from growing into "nothing can ever be deleted".
    """
    mgr, _ = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    mgr.remove_session("training")

    assert mgr.get_session("training") is None, "an explicit removal must delete the record"
    assert _stored(mgr) == {}, "and must reach disk"


@pytest.mark.parametrize(
    ("module", "stop", "action_word"),
    [
        pytest.param(
            train_mod,
            lambda: lerobot_train(dataset_root=UNUSED_DATASET, action="stop", session_name="training"),
            "train",
            id="train",
        ),
        pytest.param(
            tele_mod,
            lambda: tele_mod.lerobot_teleoperate(action="stop", session_name="training"),
            "teleoperate",
            id="teleoperate",
        ),
    ],
)
def test_stop_can_still_reach_a_session_it_could_not_inspect(
    monkeypatch: pytest.MonkeyPatch, module: Any, stop: Any, action_word: str
) -> None:
    """The operator-visible point: such a session stays stoppable, from either tool.

    Reaching it is the property pinned here - the record survives the load and
    the signals go to the recorded pid. The *verdict* cannot be affirmative: the
    same ``AccessDenied`` that hid the process from the store also hides whether
    it exited, and ``stop`` reports that as unknown rather than claiming an exit
    it could not observe. Before this rule reached one owner that report was
    unreachable for whichever tool's records the other one had already pruned.
    """
    _, pid = _seed(action=action_word)
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(module.os, "kill", lambda p, sig: signalled.append((p, sig)))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)

    result = stop()

    # psutil.pid_exists probes with signal 0, so only a real signal counts as a stop.
    sent = [entry for entry in signalled if entry[1] != 0]
    assert sent and sent[0] == (pid, signal.SIGTERM), f"stop must signal the recorded pid, sent {sent}"
    verdict = next(block["json"] for block in result["content"] if "json" in block)["stopped"]
    assert verdict is None, "an exit that could not be observed is unknown, not reported either way"
    assert "training" in _stored(SessionManager()), "the record must survive so the session stays stoppable"


def test_retaining_the_record_is_reported(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A denied probe is the operator's only clue, so it must not be silent."""
    mgr, pid = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    with caplog.at_level("WARNING"):
        mgr.list_sessions()

    assert any(str(pid) in r.getMessage() for r in caplog.records), (
        "a session that could not be inspected must be reported, naming the pid"
    )


#: Start offset a seeded record claims for its process. Any value does: what the
#: verdict reads is whether the process now holding the pid reports this one.
_RECORDED_START_S = 1.0


def _pose_a_reused_pid(mgr: Any, pid: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the live recorded pid report an identity the record does not claim.

    This is what a finished run looks like once the kernel hands its number out
    again, and it is the only spelling of "finished" that reaches
    ``session_is_running``'s identity comparison: the record has to *carry* an
    identity, or the verdict short-circuits to existence and the pid is live.

    The double sits on ``_started_since_boot`` because that is the seam the
    verdict reads - on Linux it reads ``/proc/<pid>/stat`` directly and would
    bypass a ``psutil.Process`` stand-in, and letting the real read answer would
    mismatch for a different reason than the one being posed.
    """
    mgr.add_session(
        "training",
        {"pid": pid, "action": "train", "start_time": 0.0, train_mod.PID_STARTED_SINCE_BOOT: _RECORDED_START_S},
    )
    monkeypatch.setattr(_process_stop, "_started_since_boot", lambda p: _RECORDED_START_S + 3600.0)


# ---------------------------------------------------------------------------
# A finished run is retained too, however the load path reports it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arrange", "id_"),
    [
        pytest.param(
            lambda mgr, pid, mp: mp.setattr(train_mod.psutil, "pid_exists", lambda pid: False),
            "the pid is gone",
            id="pid-absent",
        ),
        pytest.param(
            lambda mgr, pid, mp: _raise_on_probe(mp, train_mod.psutil.NoSuchProcess),
            "reaped between the two probes",
            id="no-such-process",
        ),
        pytest.param(_pose_a_reused_pid, "the pid was handed out again", id="pid-reused"),
    ],
)
def test_a_finished_run_keeps_its_record(monkeypatch: pytest.MonkeyPatch, arrange: Any, id_: str) -> None:
    """All three spellings of "finished" are one state, so all three are kept.

    This store retains a finished session on purpose, so ``status`` can still
    report the final log tail. Which spelling a given finish produces is a race
    between two probes, or the accident of whether the number has been reissued
    yet, so classifying them differently would make retention depend on timing
    rather than on the run.
    """
    mgr, pid = _seed()
    arrange(mgr, pid, monkeypatch)

    assert "training" in mgr.list_sessions(), f"a finished run must keep its record ({id_})"
    assert "training" in _stored(mgr), f"and must keep it on disk ({id_})"


# ---------------------------------------------------------------------------
# Controls: retention is not a running claim.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arrange", "id_"),
    [
        pytest.param(
            lambda mgr, pid, mp: mp.setattr(train_mod.psutil, "pid_exists", lambda pid: False),
            "whose pid is gone",
            id="pid-absent",
        ),
        pytest.param(_pose_a_reused_pid, "whose pid another process now holds", id="pid-reused"),
    ],
)
def test_a_retained_record_is_not_reported_running(monkeypatch: pytest.MonkeyPatch, arrange: Any, id_: str) -> None:
    """Keeping the record cannot be what makes a session read as running.

    This is the reason retention is safe, and it takes both rows to say so: the
    verdict is not the record's presence, and it is not the pid's existence
    either. A reused pid exists, is inspectable and is not the session, so only
    the second row separates the claim production makes - the pid still holds the
    process the record was written for - from the weaker one that it exists.
    """
    mgr, pid = _seed()
    arrange(mgr, pid, monkeypatch)

    result = lerobot_train(dataset_root=UNUSED_DATASET, action="status", session_name="training")

    text = next(block["text"] for block in result["content"] if "text" in block)
    assert "Status: Stopped" in text, f"a retained record {id_} must read as stopped: {text}"


def test_a_record_whose_pid_is_not_a_pid_is_reported_as_kept(caplog: pytest.LogCaptureFixture) -> None:
    """The one retained case where nothing can be signalled must say so.

    A ``pid`` field that is not a process id names nothing this store can probe,
    and converting it would name a different process - so the record is kept and
    the run reads as not running. That leaves the warning as the operator's only
    clue that a run may still be holding the GPU under a pid nothing here can
    name, which is why it says the record is kept rather than reporting a drop.
    """
    mgr = SessionManager()
    mgr.add_session("training", {"action": "train", "pid": "not-a-pid", "start_time": 0.0})

    with caplog.at_level("WARNING"):
        loaded = mgr.list_sessions()

    assert list(loaded) == ["training"], "the record is the only handle on the run"
    messages = [r.getMessage() for r in caplog.records if "training" in r.getMessage()]
    assert messages, "a record naming no process id must be reported"
    assert "kept" in messages[0], f"the report must not claim a drop: {messages[0]}"


def test_a_corrupt_store_still_degrades_to_empty() -> None:
    """Retention is about classification, not about tolerating a broken file."""
    mgr = SessionManager()
    mgr.sessions_file.parent.mkdir(parents=True, exist_ok=True)
    mgr.sessions_file.write_text("{not json")

    assert mgr.list_sessions() == {}


def _store_one_record_with_an_undecodable_byte(mgr: Any, pid: int) -> None:
    """Write a store whose JSON is well-formed but whose bytes are not UTF-8.

    The 0xE9 sits inside a session *name*, which is the field a hand-edit or a
    latin-1 writer touches; every other field, the pid included, stays ASCII.
    """
    mgr.sessions_file.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"run-cafe": {"pid": pid, "action": "train", "start_time": 0.0}}, indent=2)
    mgr.sessions_file.write_bytes(raw.encode("utf-8").replace(b"cafe", b"caf\xe9"))


def test_a_store_byte_that_is_not_utf8_still_yields_the_pid_it_names() -> None:
    """The one damage shape the handler cannot answer, so the read must not raise.

    ``UnicodeDecodeError`` is a ``ValueError``, so it is neither of the two
    failures ``except (OSError, JSONDecodeError)`` above was written for - the
    store is gone, or it is not JSON. A strict read therefore aborts whichever
    action consulted the store, and by the module docstring that includes
    ``stop``, which is the only supported way to end a detached run. Damage
    stays in the field that carries it instead: a pid is ASCII either way.
    """
    mgr = SessionManager()
    pid = _live_pid()
    _store_one_record_with_an_undecodable_byte(mgr, pid)

    loaded = mgr.list_sessions()

    assert [info["pid"] for info in loaded.values()] == [pid], (
        f"a byte outside the pid's field must not cost the pid: {loaded}"
    )


def test_stop_reaches_a_session_whose_record_carries_an_undecodable_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-visible point, driven through the tool rather than the store.

    The name to pass is the one ``list`` reports, since that is the only
    spelling of it the operator can see; what is pinned is that the signals
    still reach the recorded pid.
    """
    mgr = SessionManager()
    pid = _live_pid()
    _store_one_record_with_an_undecodable_byte(mgr, pid)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(train_mod.os, "kill", lambda p, sig: signalled.append((p, sig)))

    listed = lerobot_train(dataset_root=UNUSED_DATASET, action="list")
    assert listed["status"] == "success", f"list must report the store it can read: {listed}"
    name = next(iter(next(block["json"] for block in listed["content"] if "json" in block)["sessions"]))

    lerobot_train(dataset_root=UNUSED_DATASET, action="stop", session_name=name)

    # psutil.pid_exists probes with signal 0, so only a real signal counts as a stop.
    sent = [entry for entry in signalled if entry[1] != 0]
    assert sent and sent[0] == (pid, signal.SIGTERM), f"stop must signal the recorded pid, sent {sent}"


# ---------------------------------------------------------------------------
# The double has to sit where the verdict is read.
# ---------------------------------------------------------------------------
class TestTheVerdictIsControlledWhereItIsAnswered:
    """A stand-in for the probes above only counts if the verdict consults it.

    ``session_is_running`` resolves ``psutil`` from
    :mod:`strands_robots.tools._process_stop`'s own globals, so rebinding
    ``lerobot_teleoperate.psutil`` installs a stand-in the verdict never looks
    at. It then falls through to the real host and the session's reported state
    is decided by whether this machine happens to hold the pid the test named -
    a pass where it is free, a failure where it is taken, and a grade of nothing
    either way.

    Rebinding a whole module in another module's globals is the right tool when
    that module is the reader, and it usually is: of the 21 such rebindings in
    this tree, 20 are sound, 15 of them installing a fake clock. So the census
    below asks only about ``psutil``, whose answer a second module owns.
    """

    def test_no_test_reaches_the_verdict_by_rebinding_psutil(self) -> None:
        """No test controls a process probe by rebinding the name ``psutil``."""
        root = Path(__file__).resolve().parents[2]
        offenders = []
        accepted = 0
        for path in sorted((root / "tests").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setattr"
                    and len(node.args) >= 2
                ):
                    continue
                target, name = node.args[0], node.args[1]
                if isinstance(target, ast.Attribute) and target.attr == "psutil":
                    accepted += 1
                elif (
                    isinstance(target, ast.Name)
                    and isinstance(name, ast.Constant)
                    and name.value == "psutil"
                    # This module carries the one occurrence there is a reason for:
                    # the case below installs such a stand-in in order to assert
                    # that the verdict does not consult it.
                    and path.name != Path(__file__).name
                ):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
        assert accepted, "premise: some test must reach psutil for this rule to be about anything"
        assert not offenders, (
            "a stand-in installed as <module>.psutil is not consulted by the running verdict, which "
            "is answered in strands_robots.tools._process_stop; set the attribute on "
            "the psutil module object instead, and the identity read at "
            f"_process_stop._started_since_boot, as _raise_on_probe does: {offenders}"
        )

    def test_a_rebinding_in_the_tool_module_is_not_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rebinding ``tele_mod.psutil`` leaves the verdict reading the real host.

        The stand-in would call the session live - it reports the pid as existing,
        and a record carrying no identity is answered on existence alone. The
        session reads as finished anyway, and the stand-in is never asked: both
        halves say the verdict came from the host.
        """
        consulted: list[str] = []

        class _WouldCallItLive:
            NoSuchProcess = train_mod.psutil.NoSuchProcess
            AccessDenied = train_mod.psutil.AccessDenied

            @staticmethod
            def pid_exists(pid: int) -> bool:
                consulted.append("pid_exists")
                return True

            @staticmethod
            def Process(pid: int):  # noqa: N802 - mirror psutil.Process
                consulted.append("Process")
                raise _WouldCallItLive.NoSuchProcess(pid)

        mgr = SessionManager()
        record = {"pid": _free_pid(), "action": "teleoperate", "start_time": 0.0}
        mgr.add_session("unidentified", record)
        monkeypatch.setattr(tele_mod, "psutil", _WouldCallItLive)

        assert _process_stop.session_is_running(record) is False, "the stand-in would have called this live"
        assert consulted == [], f"the verdict must not be reachable this way, but consulted {consulted}"

    def test_the_module_object_is_the_seam_the_verdict_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the attribute on the shared psutil object does reach the verdict.

        The control for the census above: the rule is about how a double is
        installed, not about leaving the probes alone.
        """
        asked: list[int] = []

        def pid_exists(pid: int) -> bool:
            asked.append(int(pid))
            return False

        mgr, pid = _seed()
        monkeypatch.setattr(train_mod.psutil, "pid_exists", pid_exists)

        record = mgr.get_session("training")
        assert record is not None, "premise: the record is retained"
        assert _process_stop.session_is_running(record) is False
        assert pid in asked, f"the verdict must read the module object, but asked {asked}"
