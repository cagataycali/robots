"""One session store, one prune policy, one lock - shared by both lerobot tools.

``lerobot_teleoperate`` and ``lerobot_train`` record their detached children in
the same file on purpose, so ``list`` shows every robot session at once. That
file is the only place a detached child's pid is written down, so a lost record
leaves the child running with no supported way to stop it.

Each tool used to carry its own ``SessionManager`` over that one file, and the
copies disagreed in three ways that these tests pin:

* one pruned finished records **on every read** and wrote the pruned map back, so
  a read-only teleoperation ``list`` erased a training run's record;
* they disagreed about a record whose ``pid`` field is not a process id - one
  dropped it, the other kept it - so one file read two ways;
* both were load-modify-write with no lock, and
  :func:`~strands_robots.tools._process_stop.store_sessions` derived its temp
  file from the store's name alone, so two writers shared one temp path and could
  commit a document neither of them wrote.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools import _process_stop, _session

#: A pid no process holds, so a record naming it is provably finished.
_DEAD_PID = 999999


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the one store at a throwaway directory."""
    monkeypatch.setattr(_session, "SESSION_DIR", tmp_path / ".sessions")
    return tmp_path / ".sessions"


def _stored(mgr: Any) -> dict[str, Any]:
    """The records on disk, independent of what a load returned."""
    if not mgr.sessions_file.exists():
        return {}
    return json.loads(mgr.sessions_file.read_text())


def test_both_tools_use_the_one_session_manager() -> None:
    """Two copies of this class over one file is what let the policies drift."""
    assert tele_mod.SessionManager is train_mod.SessionManager is _session.SessionManager


@pytest.mark.parametrize("reader", ["teleop", "train"], ids=["teleop-reads", "train-reads"])
@pytest.mark.parametrize(
    ("label", "record"),
    [
        ("finished", {"pid": _DEAD_PID, "action": "train"}),
        ("unusable-pid", {"pid": "not-a-pid", "action": "train"}),
        ("pidless", {"action": "train"}),
    ],
)
def test_a_read_by_either_tool_erases_nothing(reader: str, label: str, record: dict[str, Any]) -> None:
    """A listing is a query: it must not delete the record it reports on.

    The pruning copy wrote its prune back to the shared file, so a teleoperation
    ``list`` deleted a training record - including the finished one the training
    tool retains so ``status`` can still tail its log.
    """
    writer = train_mod.SessionManager()
    writer.add_session(label, record)
    assert label in _stored(writer), "premise: the record must reach disk"

    other = (tele_mod if reader == "teleop" else train_mod).SessionManager()
    assert label in other.list_sessions(), "a read must report the record it holds"
    assert label in _stored(other), f"a {reader} read erased a record it did not write"


@pytest.mark.parametrize("tool", ["teleop", "train"], ids=["teleop", "train"])
def test_a_write_reaps_only_what_it_can_prove_is_gone(tool: str) -> None:
    """One prune policy, and it drops only a record whose process is provably gone.

    A pid nothing can inspect is not evidence a run ended, and the record is the
    only handle on it - so it is kept, whichever tool writes next.
    """
    mgr = (tele_mod if tool == "teleop" else train_mod).SessionManager()
    mgr.add_session("finished", {"pid": _DEAD_PID})
    mgr.add_session("unusable", {"pid": "not-a-pid"})
    mgr.add_session("pidless", {"action": "record"})

    mgr.add_session("live", {"pid": os.getpid()})

    assert sorted(_stored(mgr)) == ["live", "pidless", "unusable"]


def _lock_is_held(lock_file: Path) -> bool:
    """Whether some open file description already holds ``lock_file`` exclusively.

    A second :func:`open` makes a new file description, and ``flock`` locks those
    rather than processes, so this answers truthfully from inside the holder's own
    process.
    """
    with open(lock_file, "a+", encoding="utf-8") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        return False


@pytest.mark.parametrize("verb", ["add_session", "remove_session"])
def test_the_load_and_the_store_happen_inside_the_lock(verb: str) -> None:
    """Both write verbs are load-modify-write, so the whole sequence is guarded.

    Holding the lock for only the write would not help: the record that gets lost
    is the one written between another writer's load and its store, so the load
    has to be inside the lock too.
    """
    mgr = _session.SessionManager()
    mgr.add_session("seed", {"pid": os.getpid()})
    held: dict[str, bool] = {}
    real_read = _session.SessionManager._read
    real_save = _session.SessionManager._save_sessions

    def watched_read(self: Any) -> dict[str, Any]:
        held["read"] = _lock_is_held(self.lock_file)
        return real_read(self)

    def watched_save(self: Any, sessions: Any) -> None:
        held["save"] = _lock_is_held(self.lock_file)
        real_save(self, sessions)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_session.SessionManager, "_read", watched_read)
        patch.setattr(_session.SessionManager, "_save_sessions", watched_save)
        if verb == "add_session":
            mgr.add_session("later", {"pid": os.getpid()})
        else:
            mgr.remove_session("seed")

    assert held == {"read": True, "save": True}, f"{verb} left part of the sequence unguarded: {held}"


def test_the_lock_excludes_a_second_writer_while_one_holds_it() -> None:
    """The lock is a file the commit never replaces, so it really excludes.

    A lock taken on the store itself would be held on the inode
    :func:`os.replace` swaps out, and a second writer locking the new inode would
    not be excluded at all.
    """
    mgr = _session.SessionManager()
    assert mgr.lock_file != mgr.sessions_file, "a replaced file cannot carry the lock"

    entered = threading.Event()
    second_got_in = threading.Event()

    def contend() -> None:
        with _session.SessionManager()._locked():
            second_got_in.set()

    with mgr._locked():
        entered.set()
        thread = threading.Thread(target=contend)
        thread.start()
        assert not second_got_in.wait(timeout=1.0), "a second writer entered while the lock was held"
    thread.join(timeout=10)
    assert second_got_in.is_set(), "the lock must be released when the block ends"


def test_each_writer_commits_through_a_temp_file_of_its_own() -> None:
    """A temp path shared by two writers can commit a document neither wrote.

    ``json`` is written into the temp file and then renamed over the store, so two
    processes writing one temp path interleave into it and
    :func:`os.replace` commits the mixture - which the load path reads as *no
    sessions*, losing every recorded pid at once rather than one.
    """
    store = _session.session_dir() / _session.SESSIONS_FILENAME
    _session.ensure_session_dir()
    seen: list[str] = []
    real_replace = os.replace

    def record_replace(src: Any, dst: Any) -> None:
        seen.append(Path(src).name)
        real_replace(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_process_stop.os, "replace", record_replace)
        for pid in (111, 222):
            patch.setattr(_process_stop.os, "getpid", lambda pid=pid: pid)
            _process_stop.store_sessions(store, {"s": {"pid": pid}})

    assert len(set(seen)) == 2, f"two writers shared one temp path: {seen}"
    assert all("111" in seen[0] or "222" in name for name in seen), f"the temp path must name its writer: {seen}"


def test_importing_a_tool_writes_nothing_to_the_working_directory(tmp_path: Path) -> None:
    """A library that may never be called must not litter the caller's cwd.

    Run in a child process because importing here would already have happened,
    and the directory is what an import used to create.
    """
    for module in ("lerobot_teleoperate", "lerobot_train"):
        target = tmp_path / module
        target.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", f"import strands_robots.tools.{module}"],
            cwd=target,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"importing {module} failed: {result.stderr}"
        assert list(target.iterdir()) == [], f"importing {module} wrote {list(target.iterdir())} into the cwd"
