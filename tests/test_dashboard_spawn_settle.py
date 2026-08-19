"""Q1: a pid is not a running robot.

``POST /api/devices/spawn`` used to answer ``200 {"peer_id":…, "pid":1234}`` the
instant ``Popen`` returned -- which it always does. A child with a wrong camera
config, a port another process holds, or a missing policy still gets a pid, dies
a second later, and the dashboard drew a card for a peer that would never
appear. The operator walks away believing an arm is live.

These tests pin ``DeviceManager.settle()`` and ``crash_reason()``.
"""

from __future__ import annotations

from collections import deque

import pytest

from strands_robots.dashboard.device_manager import (
    DeviceManager,
    ManagedRobot,
    crash_reason,
)


class FakeProc:
    """A Popen stand-in whose exit is scripted, not timed."""

    def __init__(self, *, exits_after: int | None = None, code: int = 1, pid: int = 4242) -> None:
        self.pid = pid
        self._exits_after = exits_after
        self._code = code
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self._exits_after is not None and self.polls > self._exits_after:
            return self._code
        return None


class Clock:
    """Deterministic monotonic time: sleeping advances it, nothing waits."""

    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.001)


def _manager(**managed_kwargs) -> tuple[DeviceManager, ManagedRobot]:
    dm = DeviceManager.__new__(DeviceManager)  # no hardware probing in __init__
    m = ManagedRobot(peer_id="so101-real-1", robot_name="so101", mode="real", **managed_kwargs)
    dm.robots = {"so101-real-1": m}
    return dm, m


# --------------------------------------------------------------------------
# crash_reason: the cause, not the fallout
# --------------------------------------------------------------------------

def test_the_cause_wins_over_the_cleanup_error_that_follows_it():
    """The exact log from the live incident, in the order it was written."""
    lines = [
        "02:12:54 Traceback (most recent call last):",
        '02:12:54   File "/…/hardware_robot.py", line 177, in _build_camera_config',
        "02:12:54     raise ValueError(",
        "02:12:54 ValueError: Camera 'main' config must be a mapping of option name to value, got int: 3.",
        "02:12:54 ERROR:strands_robots.hardware_robot:Cleanup error for so101: 'Robot' object has no attribute 'robot'",
    ]

    reason = crash_reason(lines)

    assert reason is not None
    assert reason.startswith("Traceback (most recent")  # the traceback header leads the fault
    assert "Cleanup error" not in reason  # never blame teardown


def test_a_bare_exception_line_is_reported_with_its_message():
    reason = crash_reason(["10:00:00 ValueError: Camera 'main' config must be a mapping, got int: 3."])
    assert reason == "ValueError: Camera 'main' config must be a mapping, got int: 3."


def test_a_logger_prefix_is_stripped_so_the_message_leads():
    reason = crash_reason(["10:00:00 ERROR:strands_robots.mesh.core:ConnectionError: Port is in use!"])
    assert reason == "ConnectionError: Port is in use!"


def test_ordinary_startup_chatter_is_not_a_reason():
    assert crash_reason([
        "10:00:00 INFO:strands_robots:connecting so101",
        "10:00:01 INFO:strands_robots:zenoh session opened",
        "",
    ]) is None


def test_no_output_at_all_blames_nothing():
    assert crash_reason([]) is None
    assert crash_reason(["", "   "]) is None


def test_a_reason_is_capped_so_one_line_stays_one_line():
    reason = crash_reason(["RuntimeError: " + "x" * 2000])
    assert reason is not None and len(reason) <= 400


# --------------------------------------------------------------------------
# settle: the four honest answers
# --------------------------------------------------------------------------

def test_a_child_that_dies_is_reported_as_failed_with_the_reason():
    proc = FakeProc(exits_after=2, code=1)
    dm, m = _manager(process=proc)
    m.logs = deque([
        "02:12:54 ValueError: Camera 'main' config must be a mapping of option name to value, got int: 3.",
        "02:12:54 ERROR:strands_robots.hardware_robot:Cleanup error for so101: no attribute 'robot'",
    ])
    clock = Clock()

    out = dm.settle("so101-real-1", timeout=5, sleep=clock.sleep, now=clock.now)

    assert out["status"] == "failed"
    assert out["exit_code"] == 1
    assert "must be a mapping" in out["reason"]
    assert out["log_tail"]  # the evidence travels with the failure


def test_a_silent_exit_still_names_the_exit_code():
    dm, _ = _manager(process=FakeProc(exits_after=0, code=2))
    clock = Clock()

    out = dm.settle("so101-real-1", timeout=1, sleep=clock.sleep, now=clock.now)

    assert out["status"] == "failed"
    assert out["reason"] == "the process exited with code 2"


def test_running_needs_the_mesh_to_have_seen_the_peer():
    dm, _ = _manager(process=FakeProc())
    clock = Clock()
    seen: list[str] = []

    def is_up(pid: str) -> bool:
        seen.append(pid)
        return len(seen) >= 3  # announces itself on the third poll

    out = dm.settle("so101-real-1", timeout=5, is_up=is_up, sleep=clock.sleep, now=clock.now)

    assert out == {"status": "running"}
    assert clock.t < 1000.0 + 5  # returned early, did not burn the window


def test_alive_but_unannounced_is_starting_not_success():
    """Slow hardware exists; claiming either outcome would be a guess."""
    dm, _ = _manager(process=FakeProc())
    clock = Clock()

    out = dm.settle("so101-real-1", timeout=2, is_up=lambda pid: False,
                    sleep=clock.sleep, now=clock.now)

    assert out["status"] == "starting"
    assert out["waited_s"] == 2.0


def test_a_pid_alone_is_never_called_running():
    """With no presence probe, an alive child is still only 'starting'."""
    dm, _ = _manager(process=FakeProc())
    clock = Clock()

    out = dm.settle("so101-real-1", timeout=1, sleep=clock.sleep, now=clock.now)

    assert out["status"] == "starting"


def test_a_broken_presence_probe_cannot_fail_a_good_spawn():
    dm, _ = _manager(process=FakeProc())
    clock = Clock()

    def exploding(pid: str) -> bool:
        raise RuntimeError("bridge is mid-restart")

    out = dm.settle("so101-real-1", timeout=1, is_up=exploding,
                    sleep=clock.sleep, now=clock.now)

    assert out["status"] == "starting"  # degraded to unconfirmed, not failed


def test_despawned_while_watching_is_gone():
    dm, _ = _manager(process=FakeProc())
    dm.robots.clear()
    clock = Clock()

    assert dm.settle("so101-real-1", timeout=1, sleep=clock.sleep, now=clock.now) == {"status": "gone"}


def test_death_is_noticed_even_with_a_zero_timeout():
    """timeout=0 must still check once -- the common case is instant death."""
    dm, m = _manager(process=FakeProc(exits_after=0, code=1))
    m.logs = deque(["10:00:00 OSError: [Errno 16] Resource busy: '/dev/cu.usbmodem5AB0181806'"])
    clock = Clock()

    out = dm.settle("so101-real-1", timeout=0, sleep=clock.sleep, now=clock.now)

    assert out["status"] == "failed"
    assert "Resource busy" in out["reason"]


@pytest.mark.parametrize("timeout", [0.0, 0.5, 3.0])
def test_settle_always_answers_with_a_status(timeout: float):
    dm, _ = _manager(process=FakeProc())
    clock = Clock()
    out = dm.settle("so101-real-1", timeout=timeout, sleep=clock.sleep, now=clock.now)
    assert out["status"] in {"running", "starting", "failed", "gone"}
