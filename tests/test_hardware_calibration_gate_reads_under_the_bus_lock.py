"""The calibration check in ``_connect_robot`` reads the bus under the device lock.

``_connect_robot`` brings a lerobot arm up in two serial conversations. The first,
``connect()``, runs inside ``_bring_up_robot`` under ``bus_lock(self.robot)``. The
second is the calibration gate: ``self.robot.is_calibrated`` on a Feetech arm is
``bus.read_calibration()``, one ``Min_Position_Limit`` / ``Max_Position_Limit``
read per servo. It ran with no lock at all.

By then ``connect()`` has made ``is_connected`` True, which is exactly the
condition the mesh's ``hw_joints`` state probe and its camera publisher wait for
before taking the lock and reading the bus. On an SO-101 spawned from the
dashboard the two readers met on the Feetech SDK's single port handler, whose
``is_using`` flag refuses the second caller::

    Failed to read 'Min_Position_Limit' on id_=1 after 1 tries. [TxRxResult] Port is in use!

``_connect_robot`` then took the except-branch, closed the port, and reported
"the motors bus did not open" for an arm whose six servos had just answered
``broadcast_ping`` from a bare bus in the same venv. Three attempts, three
refusals, 2026-10-01.

The fake below is the shape of lerobot's ``SOFollower`` with one addition: its
``is_calibrated`` read asserts that the caller holds the device lock, the way
the real bus would fail if a second reader did not.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

pytest.importorskip("lerobot")

from strands_robots.bus_access import bus_lock  # noqa: E402
from tests._hardware_robot import hardware_robot_on  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.is_connected = False


class _ArmWhoseCalibrationReadNeedsTheLock:
    """lerobot ``SOFollower``'s surface; ``is_calibrated`` is a bus read."""

    def __init__(self) -> None:
        self.name = "so101_fake"
        self.robot_type = "so101_fake"
        self.bus = _FakeBus()
        self.cameras: dict[str, Any] = {}
        self.config = type("Cfg", (), {"cameras": {}})()
        self.calibration_reads_without_lock = 0
        self.calibration_reads = 0

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.bus.connect()

    def disconnect(self) -> None:
        self.bus.disconnect(True)

    @property
    def is_calibrated(self) -> bool:
        self.calibration_reads += 1
        lock = bus_lock(self)
        # ``RLock`` tells its owner apart from everyone else: acquiring with no wait
        # succeeds for the holder (re-entrant) and fails for anyone else while held.
        # A reader that does not hold it would collide with a probe that does.
        if not lock.acquire(blocking=False):
            self.calibration_reads_without_lock += 1
            return True
        try:
            if lock._is_owned() and not _held_by_current_thread_before(lock):
                self.calibration_reads_without_lock += 1
        finally:
            lock.release()
        return True

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}


def _held_by_current_thread_before(lock: threading.RLock) -> bool:
    """Whether the current thread held ``lock`` BEFORE the acquire just made.

    ``RLock`` exposes no public recursion count; ``_release_save`` returns
    ``(count, owner)`` and restores on ``_acquire_restore``. A count above one
    means the acquire in ``is_calibrated`` was re-entrant: the caller held it.
    """
    state = lock._release_save()  # type: ignore[attr-defined]
    try:
        count = state[0] if isinstance(state, tuple) else 1
    finally:
        lock._acquire_restore(state)  # type: ignore[attr-defined]
    return count > 1


def test_the_calibration_gate_reads_the_bus_under_the_device_lock() -> None:
    arm = _ArmWhoseCalibrationReadNeedsTheLock()
    hw = hardware_robot_on(arm, tool_name="so101_fake", control_frequency=1000.0)

    ok, err = asyncio.run(hw._connect_robot())

    assert ok, err
    assert arm.calibration_reads == 1, "the gate must read the flag exactly once"
    assert arm.calibration_reads_without_lock == 0, (
        "is_calibrated was read without bus_lock(robot): on a Feetech arm that read is a serial "
        "conversation that collides with the mesh hw_joints probe ('Port is in use!')"
    )
