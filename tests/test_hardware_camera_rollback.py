"""A failed connect must not leave a camera it already opened streaming.

``strands_robots.hardware_robot.Robot._close_open_devices`` exists so a
connect that fails partway can be retried. lerobot's robots open their devices
in sequence -- ``bus.connect()``, then ``for cam in self.cameras.values():
cam.connect()`` -- and neither the loop nor the failing camera closes the
cameras opened before it. Rolling back only the serial port therefore leaves
the camera set half-open, and that is not merely untidy: it makes the next
attempt unrecoverable.

lerobot gates both recovery paths on ``is_connected``, which is
``bus.is_connected and all(cam.is_connected ...)``:

    - ``Robot.connect()`` is ``@check_if_already_connected``, so the retry
      raises ``DeviceAlreadyConnectedError`` on the camera that is *healthy*,
      masking the camera that actually failed;
    - ``Robot.disconnect()`` is ``@check_if_not_connected``, so it refuses to
      run while any one camera is still shut.

These tests pin that the rollback closes every device it can, so the retry
keeps surfacing the real failure:

    - a camera opened before the failing one is closed;
    - the retry still reports the camera that actually failed, instead of
      degrading into a generic "Failed to connect" that names nothing;
    - a camera that never opened is left alone (it released its own resources
      in ``connect()``, and disconnecting it again would raise);
    - the cameras are closed independently, so one stuck camera cannot keep the
      rest of the set open -- the failure mode lerobot's own ``disconnect()``
      has;
    - a close that raises never masks the original connect error;
    - both call sites roll the cameras back: the explicit ``_connect_robot``
      and the lazy connect inside ``send_action``.

No camera device and no serial port is opened: the lerobot driver and its
cameras are in-memory fakes that mirror lerobot's connect ordering and its
``is_connected`` / decorator contracts.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState


class _FakeCamera:
    """Mirrors ``OpenCVCamera``'s connect/disconnect contract.

    ``connect()`` releases its own resources before re-raising (lerobot's
    ``except BaseException: self._cleanup_resources()``), so a camera that
    fails to open reports ``is_connected`` False -- which is why the rollback
    must skip it rather than disconnect it.
    """

    def __init__(self, name: str, *, fails: bool = False, close_raises: bool = False) -> None:
        self.name = name
        self.is_connected = False
        self._fails = fails
        self._close_raises = close_raises
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, warmup: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        self.connect_calls += 1
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        if self._fails:
            # lerobot's camera cleans up after itself, then re-raises.
            raise ConnectionError(f"Failed to open {self.name}.")
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} not connected.")
        if self._close_raises:
            raise OSError(f"{self.name} close failed")
        self.is_connected = False


class _FakeBus:
    def __init__(self) -> None:
        self.is_connected = False
        self.disconnect_calls: list[bool] = []

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.is_connected = False


class _FakeCameraRobot:
    """Mirrors lerobot ``SOFollower``'s connect / disconnect / is_connected.

    Connect order and the absence of any camera cleanup in the loop are the
    behaviour under test, so they are reproduced exactly.
    """

    def __init__(self, cameras: dict[str, _FakeCamera]) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.bus = _FakeBus()
        self.cameras = cameras
        self.is_calibrated = True
        self.configure_calls = 0
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": cameras})()

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - lerobot signature
        if self.is_connected:  # @check_if_already_connected
            raise DeviceAlreadyConnectedError("SOFollower is already connected.")
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure_calls += 1

    def disconnect(self) -> None:
        if not self.is_connected:  # @check_if_not_connected
            raise DeviceNotConnectedError("SOFollower is not connected. Run `.connect()` first.")
        self.bus.disconnect(True)
        for cam in self.cameras.values():
            cam.disconnect()

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


def _make_robot(fake: Any) -> HwRobot:
    """Construct a Robot around ``fake``, bypassing hardware init."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 1000.0
    hw.action_sleep_time = 0.001
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw.mesh = None
    hw.peer_id = None
    hw.robot = fake
    return hw


def _robot_with_failing_second_camera(**kwargs: Any) -> _FakeCameraRobot:
    """A two-camera arm whose second camera cannot be opened."""
    return _FakeCameraRobot(
        {
            "wrist": _FakeCamera("wrist_cam", **kwargs),
            "top": _FakeCamera("top_cam", fails=True),
        }
    )


class TestCamerasAreRolledBack:
    """A camera opened before the failing one must not stay streaming."""

    def test_a_camera_opened_before_the_failing_one_is_closed(self) -> None:
        """``connect()`` opens ``wrist`` and then fails on ``top``. The
        rollback must close ``wrist``: lerobot's connect loop does not, and
        while it stays open the device node is held and its read thread runs."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False
        assert "top_cam" in err
        assert fake.cameras["wrist"].is_connected is False
        assert fake.cameras["wrist"].disconnect_calls == 1
        hw.cleanup()

    def test_the_bus_and_the_cameras_are_both_closed(self) -> None:
        """One rollback covers the whole device set. Closing only the port
        leaves ``is_connected`` False for a different reason and the retry
        still cannot get past the open camera."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        asyncio.run(hw._connect_robot())

        assert fake.bus.is_connected is False
        assert fake.bus.disconnect_calls == [False]  # no torque write on a failed bring-up
        assert all(c.is_connected is False for c in fake.cameras.values())
        hw.cleanup()

    def test_a_camera_that_never_opened_is_not_disconnected(self) -> None:
        """The failing camera released its own resources inside ``connect()``,
        so it reports ``is_connected`` False. Disconnecting it anyway would
        raise ``DeviceNotConnectedError`` from the rollback path."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        asyncio.run(hw._connect_robot())

        assert fake.cameras["top"].disconnect_calls == 0
        hw.cleanup()


class TestTheRetryStillDiagnosesTheRealFault:
    """The point of the rollback: the next attempt reports the same fault."""

    def test_the_retry_still_names_the_camera_that_actually_failed(self) -> None:
        """With ``wrist`` left open, the retry's ``connect()`` raises
        ``DeviceAlreadyConnectedError`` on ``wrist``; ``_connect_robot``
        tolerates any "already connected" message, so the error degrades to a
        generic "Failed to connect" that names no device at all -- and stays
        that way for every later attempt. Each attempt must instead keep
        naming ``top_cam``."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        errors = [asyncio.run(hw._connect_robot())[1] for _ in range(3)]

        assert all("top_cam" in err for err in errors), errors
        # the connect was genuinely retried each time, not short-circuited
        assert fake.cameras["top"].connect_calls == 3
        hw.cleanup()

    def test_the_arm_is_never_left_reporting_a_configured_bus(self) -> None:
        """``configure()`` runs after the camera loop, so it never executed.
        A retry that is swallowed as "already connected" would leave the bus
        open with its operating-mode and PID registers unwritten."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        for _ in range(3):
            asyncio.run(hw._connect_robot())

        assert fake.configure_calls == 0
        assert fake.bus.is_connected is False
        hw.cleanup()


class TestRollbackIsBestEffortPerDevice:
    """A rollback runs from an ``except`` handler: it must not raise, and one
    device must not block the others."""

    def test_one_stuck_camera_does_not_keep_the_rest_of_the_set_open(self) -> None:
        """This is the failure mode lerobot's own ``disconnect()`` has: a
        single loop with no per-device guard, so the first close that raises
        abandons every camera after it."""
        stuck = _FakeCamera("stuck_cam", close_raises=True)
        healthy = _FakeCamera("healthy_cam")
        fake = _FakeCameraRobot({"stuck": stuck, "healthy": healthy, "top": _FakeCamera("top_cam", fails=True)})
        hw = _make_robot(fake)

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False
        assert "top_cam" in err  # original fault, not the stuck close
        assert healthy.is_connected is False  # closed despite the earlier raise
        assert stuck.disconnect_calls == 1  # attempted
        hw.cleanup()

    def test_a_camera_close_that_raises_does_not_mask_the_connect_error(self) -> None:
        """The caller must see why the connect failed, not why the cleanup
        failed."""
        fake = _robot_with_failing_second_camera(close_raises=True)
        hw = _make_robot(fake)

        ok, err = asyncio.run(hw._connect_robot())

        assert ok is False
        assert "top_cam" in err
        assert "close failed" not in err
        hw.cleanup()

    def test_a_non_mapping_cameras_attribute_still_rolls_the_bus_back(self) -> None:
        """lerobot's contract is ``dict[str, Camera]``. A driver exposing
        something else is not a camera set the rollback can walk, but that must
        not stop the serial port from being closed."""
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)
        fake.cameras = ["not", "a", "mapping"]  # type: ignore[assignment]

        ok, _err = asyncio.run(hw._connect_robot())

        assert ok is False
        assert fake.bus.is_connected is False
        assert fake.bus.disconnect_calls == [False]
        hw.cleanup()


class TestLazyTeleopConnectRollsCamerasBackToo:
    """``send_action`` lazily connects on its first call and rolls back on
    failure; it must roll the cameras back for the same reason."""

    def test_send_action_closes_a_camera_opened_by_a_failed_lazy_connect(self) -> None:
        fake = _robot_with_failing_second_camera()
        hw = _make_robot(fake)

        r1 = hw.send_action({"j0.pos": 0.1})
        r2 = hw.send_action({"j0.pos": 0.2})

        assert r1["status"] == "error"
        assert r2["status"] == "error"
        # each call retried the connect and kept reporting the real fault
        assert fake.cameras["top"].connect_calls == 2
        assert "top_cam" in r2["content"][0]["text"]
        assert fake.cameras["wrist"].is_connected is False
        assert fake.sent_actions == []  # nothing written to an unbuilt arm
        hw.cleanup()
