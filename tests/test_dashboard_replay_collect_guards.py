"""Q4: two clicks used to orphan a process.

``replay()`` and ``collect()`` minted their peer id as
``f"replay-{int(time.time()) % 100000}"``, so two starts in the SAME SECOND
produced the same id and the second one's ``self.robots[peer_id] = managed``
overwrote the first's tracking entry. The first process then existed but was
unreachable -- no logs, no despawn, no way to stop it -- while still publishing
to the mesh under an id that now belonged to someone else.

There was also no already-running guard, so clicking Run twice started two sims
of the same episode, and starting two recordings pointed two writers at one
dataset directory.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

import strands_robots.dashboard.device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager, ManagedRobot


class FakeProc:
    """Popen stand-in whose liveness is a flag."""

    _next_pid = 5000

    def __init__(self) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.stdout = None
        self.dead = False

    def poll(self):
        return 0 if self.dead else None


@pytest.fixture
def manager(monkeypatch):
    """A DeviceManager that spawns fakes, with the clock frozen to one second.

    The stand-ins replace the NAMES inside device_manager, not attributes on the
    real modules: ``monkeypatch.setattr(dm.threading, "Thread", ...)`` mutates
    the shared threading module, and this test file's own threads then come back
    as stubs with no ``.join`` -- which is exactly how the first version of this
    fixture broke its own concurrency tests.
    """
    monkeypatch.setattr(
        dm,
        "subprocess",
        types.SimpleNamespace(Popen=lambda *a, **k: FakeProc(), PIPE=-1, STDOUT=-2),
    )
    monkeypatch.setattr(
        dm,
        "time",
        types.SimpleNamespace(
            time=lambda: 1_787_136_722.0,  # the same second, always
            monotonic=time.monotonic,
            sleep=time.sleep,
            strftime=time.strftime,
        ),
    )
    d = DeviceManager.__new__(DeviceManager)
    d.robots = {}
    d._lock = threading.Lock()
    return d


# --------------------------------------------------------------------------
# the orphan: two starts in one second
# --------------------------------------------------------------------------


def test_two_replays_in_the_same_second_are_two_tracked_peers(manager):
    """The original bug: the second start erased the first from self.robots."""
    first = manager.replay("user/ds", episode=0)
    second = manager.replay("user/ds", episode=1)  # different episode, same second

    assert first["peer_id"] != second["peer_id"]
    assert set(manager.robots) == {first["peer_id"], second["peer_id"]}
    # Both processes remain reachable: logs, despawn and stop all need this.
    assert manager.robots[first["peer_id"]].process.pid != manager.robots[second["peer_id"]].process.pid


def test_the_suffix_is_readable_not_random(manager):
    first = manager.replay("user/ds", episode=0)
    second = manager.replay("user/ds", episode=1)
    assert second["peer_id"] == f"{first['peer_id']}-2"


def test_a_dead_but_tracked_id_is_not_reused(manager):
    """Its logs are the record of what happened, and the mesh may still hold it."""
    first = manager.replay("user/ds", episode=0)
    manager.robots[first["peer_id"]].process.dead = True

    second = manager.replay("user/ds", episode=1)

    assert second["peer_id"] != first["peer_id"]
    assert first["peer_id"] in manager.robots  # history survives


def test_ids_stay_unique_under_a_thundering_herd(manager):
    """Concurrent starts must not collide even inside one second."""
    got: list[str] = []
    lock = threading.Lock()

    def start(ep: int) -> None:
        r = manager.replay("user/ds", episode=ep)
        with lock:
            got.append(r["peer_id"])

    threads = [threading.Thread(target=start, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(got) == 12
    assert len(set(got)) == 12  # no id handed out twice
    assert len(manager.robots) == 12  # nothing orphaned


# --------------------------------------------------------------------------
# the guard: the same job twice
# --------------------------------------------------------------------------


def test_the_same_episode_twice_is_refused_and_names_the_live_peer(manager):
    first = manager.replay("user/ds", episode=3)

    again = manager.replay("user/ds", episode=3)

    assert again["already_running"] is True
    assert again["peer_id"] == first["peer_id"]  # so the UI can point at that card
    assert "already replaying" in again["error"]
    assert len(manager.robots) == 1  # no second process was started


def test_a_different_episode_of_the_same_dataset_is_allowed(manager):
    manager.replay("user/ds", episode=0)
    other = manager.replay("user/ds", episode=1)
    assert other.get("already_running") is None
    assert len(manager.robots) == 2


def test_replaying_again_is_allowed_once_the_first_has_finished(manager):
    """A one-shot replay exits by design; the guard must not outlive it."""
    first = manager.replay("user/ds", episode=0)
    manager.robots[first["peer_id"]].process.dead = True

    again = manager.replay("user/ds", episode=0)

    assert again.get("already_running") is None
    assert again["peer_id"] != first["peer_id"]


def test_two_recorders_never_share_one_dataset_directory(manager):
    """This guard protects DATA: two writers interleave episodes into one file."""
    first = manager.collect("/tmp/ds-a")

    again = manager.collect("/tmp/ds-a")

    assert again["already_running"] is True
    assert again["peer_id"] == first["peer_id"]
    assert "already writing" in again["error"]
    assert len(manager.robots) == 1


def test_recording_into_a_different_directory_is_allowed(manager):
    manager.collect("/tmp/ds-a")
    other = manager.collect("/tmp/ds-b")
    assert other.get("already_running") is None
    assert len(manager.robots) == 2


def test_a_collect_reserves_its_id_before_the_process_exists(manager):
    """The reservation is what makes the guard safe under concurrency."""
    got: list[str] = []
    lock = threading.Lock()

    def start(i: int) -> None:
        r = manager.collect(f"/tmp/ds-{i}")
        with lock:
            got.append(r["peer_id"])

    threads = [threading.Thread(target=start, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(set(got)) == 8
    assert all(manager.robots[pid].process is not None for pid in got)


def test_a_replay_and_a_collect_do_not_guard_each_other(manager):
    """Different modes are different work; only like-for-like is a conflict."""
    manager.replay("user/ds", episode=0)
    c = manager.collect("/tmp/ds-a")
    assert c.get("already_running") is None
    assert len(manager.robots) == 2


def test_the_job_is_recorded_so_the_guard_has_something_to_compare(manager):
    r = manager.replay("user/ds", episode=7)
    assert manager.robots[r["peer_id"]].job == {"repo_id": "user/ds", "episode": 7}
    c = manager.collect("/tmp/ds-z")
    assert manager.robots[c["peer_id"]].job == {"dataset_root": "/tmp/ds-z"}


def test_an_untracked_manager_still_mints_the_plain_id(manager):
    """No collision, no suffix: the readable id is the common case."""
    assert manager.replay("user/ds", episode=0)["peer_id"] == "replay-36722"


def test_a_job_field_absent_from_a_peer_never_matches(manager):
    """A hand-made peer with no job must not be mistaken for a running replay."""
    manager.robots["hand-made"] = ManagedRobot(
        peer_id="hand-made",
        robot_name="so101",
        mode="replay",
        process=FakeProc(),
    )
    r = manager.replay("user/ds", episode=0)
    assert r.get("already_running") is None
