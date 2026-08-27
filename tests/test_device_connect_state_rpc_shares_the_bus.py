"""The Device Connect state RPC reads joints through the shared motor bus.

``RobotDeviceDriver.getState`` is the "what is this arm doing" RPC an operator or
an agent calls at will. It reached the wrapped lerobot device directly --
``inner = getattr(self._robot, "robot", None)`` and then
``asyncio.to_thread(inner.get_observation)`` -- so it took no bus lock and read
the FULL observation. Two independent failures followed, and the handler around
the read logs at debug and returns what it has, so both surfaced as a successful
RPC whose response simply has no ``joints`` key:

* it barged in on a reader already holding the bus and the SDK answered
  ``[TxRxResult] Port is in use!``;
* lerobot's ``get_observation`` sync-reads the motors FIRST and only then loops
  the cameras, so one dead USB camera threw away the joint positions already in
  hand -- the eleven-hour incident ``bus_access.read_joints`` exists to prevent.

The fakes for the second case are imported from the test module that pins that
incident, so this file grades the RPC against the same arm state.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("device_connect_edge")

from strands_robots.bus_access import bus_lock  # noqa: E402
from strands_robots.device_connect.robot_driver import RobotDeviceDriver  # noqa: E402

from .test_bus_read_joints import POSITIONS, _Arm, _Bus  # noqa: E402

PORT_IN_USE = (
    "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 tries. [TxRxResult] Port is in use!"
)


class _Wrapper:
    """A ``Robot``-shaped host holding one device.

    Deliberately not a ``MagicMock``: on a mock, ``inner.bus`` auto-creates a
    child that HAS ``sync_read``, and ``bus_lock`` finds an auto-created
    attribute where the real device has none -- so a mock cannot tell a lock
    that serialises from one that does not.
    """

    def __init__(self, device: Any) -> None:
        self.robot = device
        self._task_state = None
        self.tool_name_str = "so101"


class _ContendedArm:
    """An arm that refuses an overlapping transaction, exactly like the SDK."""

    def __init__(self) -> None:
        self.busy = False
        self.reads = 0
        self.refusals = 0
        self.bus = self
        self.config = type("_Cfg", (), {"num_read_retries": 3})()
        self.is_connected = True

    def _enter(self) -> None:
        if self.busy:
            self.refusals += 1
            raise ConnectionError(PORT_IN_USE)

    def sync_read(self, register: str, num_retry: int | None = None) -> dict[str, float]:  # noqa: ARG002
        self._enter()
        self.reads += 1
        return dict(POSITIONS)

    def get_observation(self) -> dict[str, float]:
        self._enter()
        self.reads += 1
        return {f"{motor}.pos": value for motor, value in POSITIONS.items()}


def _state(device: Any) -> dict[str, Any]:
    return asyncio.run(RobotDeviceDriver(_Wrapper(device)).getState())


class TestTheJointsSurviveADeadCamera:
    """The incident state: motors answer, a camera on the same driver does not."""

    def test_the_joints_are_reported_while_get_observation_raises(self) -> None:
        arm = _Arm(_Bus(POSITIONS))

        state = _state(arm)

        assert "joints" in state, (
            "the RPC reported no joints for an arm whose motors answered: "
            f"keys={sorted(state)} (get_observation raised, and the joints were already in hand)"
        )
        assert state["joints"]["shoulder_pan.pos"] == pytest.approx(1.0)
        assert len(state["joints"]) == len(POSITIONS)

    def test_the_reported_joints_are_the_arms_own_positions(self) -> None:
        arm = _Arm(_Bus(POSITIONS))

        joints = _state(arm)["joints"]

        assert joints == {f"{motor}.pos": pytest.approx(value) for motor, value in POSITIONS.items()}


class TestItWaitsItsTurnOnTheBus:
    """A reader that barges in produces a collision and no data for anyone."""

    def test_it_does_not_collide_with_a_reader_holding_the_lock(self) -> None:
        arm = _ContendedArm()
        released = threading.Event()

        def hold() -> None:
            with bus_lock(arm):
                arm.busy = True
                time.sleep(0.35)
                arm.busy = False
            released.set()

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        time.sleep(0.10)  # the holder owns the lock and the wire is in use

        state = _state(arm)
        released.wait(3)
        holder.join(3)

        assert arm.refusals == 0, f"the device refused {arm.refusals} overlapping transaction(s)"
        assert "joints" in state, f"the RPC reported no joints while another reader held the bus: keys={sorted(state)}"
        assert arm.reads == 1

    def test_the_read_happens_under_the_device_lock(self) -> None:
        """Probed from another thread: the lock is an RLock, so asking inside proves nothing."""
        bus = _Bus(POSITIONS)
        arm = _Arm(bus)
        bus.owner = arm

        _state(arm)

        assert bus.locked_during_read is True, (
            "the wire was driven with the device lock free, so a concurrent reader can collide with this RPC"
        )


class TestNothingElseChanges:
    """What the change does not claim, and what must keep working."""

    def test_a_healthy_arm_reports_the_same_joints(self) -> None:
        """With nothing contending and every camera alive, the answer is what it was."""
        arm = _ContendedArm()

        assert _state(arm)["joints"] == {f"{m}.pos": pytest.approx(v) for m, v in POSITIONS.items()}

    def test_the_frame_filter_still_applies_on_the_no_bus_fallback(self) -> None:
        """A driver exposing no readable bus falls back to the full observation."""

        class _NoBus:
            bus = None
            config = type("_Cfg", (), {"num_read_retries": 3})()

            def get_observation(self) -> dict[str, Any]:
                return {"shoulder_pan.pos": 1.0, "front": np.zeros((2, 2, 3), np.uint8)}

        state = _state(_NoBus())

        assert state["joints"] == {"shoulder_pan.pos": pytest.approx(1.0)}, (
            "a camera frame reached the joints map, so the frame filter is no longer doing its job"
        )

    def test_a_host_with_no_device_reports_no_joints_and_no_error(self) -> None:
        state = asyncio.run(RobotDeviceDriver(_Wrapper(None)).getState())

        assert "joints" not in state
        assert state == {}

    def test_the_task_state_keys_are_untouched(self) -> None:
        wrapper = _Wrapper(_ContendedArm())
        wrapper._task_state = type(
            "_Task",
            (),
            {"status": type("_S", (), {"value": "running"})(), "instruction": "pick up the cube", "step_count": 42},
        )()

        state = asyncio.run(RobotDeviceDriver(wrapper).getState())

        assert state["task_status"] == "running"
        assert state["instruction"] == "pick up the cube"
        assert state["step_count"] == 42
