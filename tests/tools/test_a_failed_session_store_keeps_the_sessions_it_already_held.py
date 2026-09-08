# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session store that cannot be written keeps the sessions it already held.

Both session tools persist a whole document to the same file: ``add_session``
and ``remove_session`` load every record, change one entry, and store the map
back. Encoding that document straight into its own destination made a rejected
write destructive - the stream is truncated first, so a full disk left a prefix
where the store was, and both load paths report an unparseable store as *no
sessions*.

The records in that file are the only place a detached process's pid is written
down, which ``SessionManager`` says out loud: it "never deletes one" because of
it, and its load path keeps a record it could not inspect "so the session stays
stoppable". One failed write dropped all of them at once, so three live
processes - one of them a teleoperation loop driving a follower arm - kept
running while ``list`` answered ``status="success"`` with no sessions and
``stop`` answered that the session was not found.

These cells pin the commit instead: the document is serialized before the
destination is touched, committed through a temp file plus ``os.replace``, and a
failed write leaves every session already recorded stored, listed and stoppable.
"""

from __future__ import annotations

import ast
import builtins
import errno
import importlib
import inspect
import io
import json
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.tools._process_stop import (
    PID_STARTED_SINCE_BOOT,
    process_started_since_boot,
)

tele_mod = importlib.import_module("strands_robots.tools.lerobot_teleoperate")
train_mod = importlib.import_module("strands_robots.tools.lerobot_train")

_REAL_OPEN = io.open

#: The store both tools write. Its ``.tmp`` commit sibling shares the prefix.
_STORE_PREFIX = "active_sessions"


class _FullDisk:
    """A text file that accepts ``allowance`` characters, then reports ENOSPC.

    A full disk fails the write that exhausts it *after* the bytes before it
    landed, which is what makes an in-place rewrite destructive and a temp-file
    commit harmless. Everything else delegates, so the file the implementation
    chose to open is the file that fills up.
    """

    def __init__(self, fh: Any, allowance: int) -> None:
        self._fh = fh
        self._left = allowance

    def write(self, data: str) -> int:
        if len(data) <= self._left:
            self._left -= len(data)
            return self._fh.write(data)
        if self._left:
            self._fh.write(data[: self._left])
            self._left = 0
        raise OSError(errno.ENOSPC, "No space left on device")

    def __enter__(self) -> _FullDisk:
        return self

    def __exit__(self, *exc: object) -> None:
        self._fh.close()

    def close(self) -> None:
        self._fh.close()

    def flush(self) -> None:
        self._fh.flush()


@pytest.fixture
def full_disk(monkeypatch: pytest.MonkeyPatch):
    """Fill the filesystem under any file whose name starts with a prefix.

    Both spellings are covered on purpose - the store itself and any ``.tmp``
    sibling a commit writes - so the injection describes the disk rather than
    the implementation, and hits whichever file is written.
    """

    def _install(prefix: str = _STORE_PREFIX, allowance: int = 120) -> None:
        def _open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            fh = _REAL_OPEN(file, mode, *args, **kwargs)
            if "w" in mode and Path(str(file)).name.startswith(prefix):
                return _FullDisk(fh, allowance)
            return fh

        # ``Path.write_text`` reaches ``io.open``; a plain ``open(...)`` call
        # reaches ``builtins.open``. Both are the disk.
        monkeypatch.setattr(io, "open", _open)
        monkeypatch.setattr(builtins, "open", _open)

    return _install


@pytest.fixture
def managers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Both session managers, pointed at one throwaway store.

    Both modules compute ``SESSION_DIR`` at import time from ``cwd``; rebind it
    so the tools share a temporary store, which is what they do in production -
    ``lerobot_train`` reuses the teleoperate directory on purpose.
    """
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return {"teleop": tele_mod.SessionManager(), "train": train_mod.SessionManager()}


@pytest.fixture
def live_pids() -> Any:
    """Hand out real, live process ids and reap them however the test ends."""
    procs: list[subprocess.Popen[bytes]] = []

    def _spawn() -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(["sleep", "60"])
        procs.append(proc)
        return proc

    yield _spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _record(pid: int, action: str) -> dict[str, Any]:
    """A session record carrying the identity the load path checks."""
    return {
        "pid": pid,
        PID_STARTED_SINCE_BOOT: process_started_since_boot(pid),
        "action": action,
        "start_time": time.time(),
    }


def _seed(managers: dict[str, Any], live_pids: Any) -> dict[str, int]:
    """Record one live teleoperation session and one live training session."""
    teleop_proc, train_proc = live_pids(), live_pids()
    managers["teleop"].add_session("teleop_arm", _record(teleop_proc.pid, "teleoperate"))
    managers["train"].add_session("train_act", _record(train_proc.pid, "train"))
    return {"teleop_arm": teleop_proc.pid, "train_act": train_proc.pid}


@pytest.mark.parametrize("writer", ["teleop", "train"])
def test_a_failed_store_keeps_every_session_already_recorded(
    managers: dict[str, Any], live_pids: Any, full_disk: Any, writer: str
) -> None:
    """A write that runs out of disk leaves the stored sessions readable.

    Parametrized over which tool performs the failing write because both write
    the same file: a training run that fills the disk must not drop the record
    of a teleoperation loop, and the reverse.
    """
    seeded = _seed(managers, live_pids)
    store = managers["teleop"].sessions_file
    before = store.read_bytes()

    full_disk()
    managers[writer].add_session("late_session", _record(live_pids().pid, "train"))

    after = store.read_bytes()
    assert json.loads(after), f"the store no longer parses, so every session in it is lost: {after!r}"
    # Both managers read the same file, so both must still see both records.
    for manager in managers.values():
        listed = manager.list_sessions()
        for name, pid in seeded.items():
            assert name in listed, f"{name} was dropped by a write that failed for another session"
            assert manager.get_session(name)["pid"] == pid
    assert json.loads(before).keys() == json.loads(after).keys()


def test_the_stop_verb_still_finds_the_session_after_a_failed_store(
    managers: dict[str, Any], live_pids: Any, full_disk: Any
) -> None:
    """The tool can still stop a live session whose store write was not the one that failed.

    The store is the only route by which ``stop`` can name the process, so this
    drives the real verb rather than the manager: a session that cannot be found
    is a process that keeps driving the follower arm.
    """
    seeded = _seed(managers, live_pids)

    full_disk()
    managers["train"].add_session("late_session", _record(live_pids().pid, "train"))

    result = tele_mod.lerobot_teleoperate(action="stop", session_name="teleop_arm")
    assert result["status"] == "success", result["content"][0]["text"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not process_started_since_boot(seeded["teleop_arm"]):
            break
        time.sleep(0.05)
    assert process_started_since_boot(seeded["teleop_arm"]) is None, "the stopped process is still running"


def test_a_record_json_cannot_encode_is_refused_before_the_store_is_touched(
    managers: dict[str, Any], live_pids: Any
) -> None:
    """A NumPy scalar in a record leaves the stored sessions byte-identical.

    ``json.dump`` encodes into the stream it is given, so a value it cannot
    encode used to raise only after a prefix of the new document had replaced
    the stored one - and past the ``OSError`` handler, which does not cover it.
    """
    _seed(managers, live_pids)
    store = managers["teleop"].sessions_file
    before = store.read_bytes()

    poisoned = _record(live_pids().pid, "train")
    poisoned["steps"] = np.int32(20000)
    with pytest.raises(ValueError) as excinfo:
        managers["train"].add_session("late_session", poisoned)

    assert "int32" in str(excinfo.value.__cause__), "the refusal must name the offending type"
    assert store.read_bytes() == before, "the stored sessions were touched by a rejected write"
    # Serializing first is what makes the refusal leave nothing at all. Encoding
    # into the temp file would keep the store intact but strand a partial one
    # beside it, and raise the encoder's own ``TypeError`` rather than a refusal
    # naming the store.
    strays = [q.name for q in store.parent.iterdir() if q.name != store.name]
    assert strays == [], f"a rejected record left a partial store behind: {strays}"


def test_a_failed_store_leaves_no_partial_sibling_behind(
    managers: dict[str, Any], live_pids: Any, full_disk: Any
) -> None:
    """A commit that did not happen leaves no second, partial store next to the real one."""
    _seed(managers, live_pids)
    store = managers["teleop"].sessions_file

    full_disk()
    managers["teleop"].add_session("late_session", _record(live_pids().pid, "teleoperate"))

    strays = [p.name for p in store.parent.iterdir() if p.name != store.name]
    assert strays == [], f"a partial store was left beside the real one: {strays}"


def test_an_unwritable_store_is_reported_and_not_raised(
    managers: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """CONTROL: an I/O failure still logs and returns rather than raising.

    Holds either way - it is the contract the tools already had, kept so the
    commit change is about durability and not about how a failure is reported.
    """
    manager = managers["teleop"]
    manager.sessions_file = manager.sessions_file.parent / "missing_dir" / "active_sessions.json"
    with caplog.at_level("ERROR"):
        manager._save_sessions({"s": {"pid": 1}})
    assert any("Error saving sessions" in r.message for r in caplog.records)


def test_a_store_that_succeeds_is_readable_by_both_managers(managers: dict[str, Any], live_pids: Any) -> None:
    """CONTROL: the ordinary path still writes a store both tools read.

    Holds either way, so a commit that refused every write would not satisfy it.
    """
    seeded = _seed(managers, live_pids)
    assert set(managers["teleop"].list_sessions()) == set(seeded)
    assert set(managers["train"].list_sessions()) == set(seeded)
    managers["teleop"].remove_session("teleop_arm")
    assert set(managers["train"].list_sessions()) == {"train_act"}


def test_both_session_stores_commit_through_the_one_owner() -> None:
    """Neither tool spells the commit itself, so the sequence lives in one place.

    Graded on the call rather than the text: the rule is which function performs
    the write, and a second copy of ``tmp`` plus ``os.replace`` is exactly the
    drift that leaves one of the two stores destructive.
    """
    for module in (tele_mod, train_mod):
        source = inspect.getsource(module.SessionManager._save_sessions)
        called = {
            ast.unparse(node.func)
            for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Call)
        }
        assert "store_sessions" in called, f"{module.__name__} does not commit through the shared owner: {called}"
        assert "json.dump" not in called, f"{module.__name__} still encodes into its own destination"
