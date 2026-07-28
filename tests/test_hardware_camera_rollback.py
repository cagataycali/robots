# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A camera that fails once at connect must not brick connect forever.

``_rollback_half_open_connect`` only touched ``self.robot.bus`` - it never
disconnected ``self.robot.cameras``. ``SOFollower.connect()`` iterates cameras in
dict order, so when the SECOND camera fails the FIRST is left
``is_connected=True``. On the next ``_connect_robot()``, ``cam.connect()`` on the
already-open camera raises ``DeviceAlreadyConnectedError``
(``OpenCVCamera.connect`` is ``@check_if_already_connected``), which the handler
swallowed as "was already connected" - so the loop NEVER reached the failing
camera again.

Measured pre-fix across three attempts with a camera that fails exactly once::

    attempt 1: ok=False bus.connect wrist.connect front.connect bus.disconnect
               state bus=False wrist=True front=False
    attempt 2: ok=False bus.connect wrist.connect       <- dies on stale wrist
    attempt 3: ok=False bus.connect wrist.connect
    front.connect() called exactly ONCE -> healthy hardware never retried

So the robot could never be connected again, even after the operator replugged
the cable, until the process restarted. The recovery route was closed too:
``SOFollower.disconnect`` is ``@check_if_not_connected``, so it raises
``DeviceNotConnectedError`` in exactly this half-open state - it cannot be the
cleanup path for the state it refuses to describe.

This is a live failure on this rig: ``molmoact2_pickplace.py`` records the front
cam USB port dropping under load ("Cannot enable. Maybe the USB cable is bad?"),
and both live scripts bypass ``_connect_robot`` entirely via
``robot.robot.connect(calibrate=False)`` - that bypass is the workaround.

The fix closes every half-open camera during rollback, and accepts an "already
connected" error only when ``robot.is_connected`` is actually True - otherwise a
subcomponent is stale-open and the correct response is rollback + retry, not
reporting success.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("lerobot")

from lerobot.utils.errors import DeviceAlreadyConnectedError  # noqa: E402

import strands_robots.hardware_robot as hardware_robot  # noqa: E402
from strands_robots.hardware_robot import Robot  # noqa: E402


class _Camera:
    """Mimics OpenCVCamera: connect is @check_if_already_connected."""

    def __init__(self, name: str, *, start_connected: bool = False, fail_times: int = 0) -> None:
        self.name = name
        self._connected = start_connected
        self.fail_times = fail_times
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, warmup: bool = True) -> None:
        self.connect_calls += 1
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError(f"{self.name}: Cannot enable. Maybe the USB cable is bad?")
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


class _UndisconnectableCamera(_Camera):
    """A camera whose disconnect() raises - must not block its siblings."""

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        raise RuntimeError("disconnect blew up")


class _StringErrorCamera(_Camera):
    """Raises the STRING form of the already-connected error, not the class."""

    def connect(self, warmup: bool = True) -> None:
        self.connect_calls += 1
        if self._connected:
            raise RuntimeError(f"{self.name} is already connected")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("bad cable")
        self._connected = True


class _Bus:
    def __init__(self) -> None:
        self._connected = False
        self.disconnect_calls = 0

    def connect(self, handshake: bool = True) -> None:
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls += 1
        self._connected = False


class _Follower:
    """Mimics SOFollower: connect opens bus then cameras; is_connected is all-or-nothing."""

    def __init__(self, **cameras: _Camera) -> None:
        self.bus = _Bus()
        self.cameras = dict(cameras)

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for camera in self.cameras.values():
            camera.connect()

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True


def _robot(follower) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.robot = follower
    hw._shutdown_event = threading.Event()
    # __new__ bypasses __init__, and __del__ -> cleanup() reads these. Set the
    # same minimal attribute set the sibling hardware tests use so the finalizer
    # is quiet; nothing here drives a task.
    hw._task_state = hardware_robot.RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm")
    hw.mesh = None
    hw.peer_id = None
    hw._stop_requested = threading.Event()
    hw._task_lock = threading.Lock()
    return hw


def _connect(hw: Robot) -> tuple[bool, str]:
    return asyncio.run(hw._connect_robot())


def _fully_closed(follower) -> bool:
    return not follower.bus.is_connected and not any(c.is_connected for c in follower.cameras.values())


class TestConnectRecoversAfterACameraFails:
    def test_a_failed_connect_leaves_no_camera_open(self):
        """The regression: the first camera stayed open and blocked every retry."""
        follower = _Follower(wrist=_Camera("wrist"), front=_Camera("front", fail_times=1))
        hw = _robot(follower)

        ok, error = _connect(hw)

        assert ok is False, error
        assert _fully_closed(follower), (
            f"bus={follower.bus.is_connected} "
            f"wrist={follower.cameras['wrist'].is_connected} "
            f"front={follower.cameras['front'].is_connected}"
        )

    def test_the_next_attempt_retries_the_failing_camera(self):
        """Pre-fix front.connect() was called exactly ONCE across three attempts."""
        follower = _Follower(wrist=_Camera("wrist"), front=_Camera("front", fail_times=1))
        hw = _robot(follower)

        assert _connect(hw)[0] is False
        first_attempt_calls = follower.cameras["front"].connect_calls

        ok, error = _connect(hw)

        assert ok is True, error
        assert follower.cameras["front"].connect_calls > first_attempt_calls, "the failing camera was never retried"
        assert follower.is_connected

    def test_recovery_works_on_a_three_attempt_sequence(self):
        """A camera that flaps twice must still recover, not accumulate staleness."""
        follower = _Follower(wrist=_Camera("wrist"), front=_Camera("front", fail_times=2))
        hw = _robot(follower)

        assert _connect(hw)[0] is False
        assert _connect(hw)[0] is False
        assert _fully_closed(follower), "state leaked across two failures"

        assert _connect(hw)[0] is True
        assert follower.is_connected

    def test_an_already_connected_robot_still_short_circuits(self):
        """The benign case must stay benign: no rollback, no reconnect churn."""
        follower = _Follower(wrist=_Camera("wrist", start_connected=True))
        follower.bus.connect()
        hw = _robot(follower)
        calls_before = follower.cameras["wrist"].connect_calls

        ok, error = _connect(hw)

        assert ok is True, error
        assert follower.cameras["wrist"].connect_calls == calls_before
        assert follower.bus.disconnect_calls == 0, "a healthy robot was rolled back"


class TestAlreadyConnectedIsNotBlanketBenign:
    def test_a_stale_open_camera_is_not_reported_as_success(self):
        """DeviceAlreadyConnectedError while NOT fully connected means half-open."""
        follower = _Follower(
            wrist=_Camera("wrist", start_connected=True),
            front=_Camera("front", fail_times=99),
        )
        hw = _robot(follower)

        ok, error = _connect(hw)

        assert ok is False, "a half-open device reported a successful connect"
        assert error
        assert _fully_closed(follower)

    def test_the_string_form_of_the_error_is_treated_the_same(self):
        """Some drivers raise a bare Exception whose text says 'already connected'."""
        follower = _Follower(
            wrist=_StringErrorCamera("wrist", start_connected=True),
            front=_StringErrorCamera("front", fail_times=99),
        )
        hw = _robot(follower)

        ok, _ = _connect(hw)

        assert ok is False
        assert not any(c.is_connected for c in follower.cameras.values())

    def test_the_error_names_the_stale_cameras(self, caplog):
        follower = _Follower(
            wrist=_Camera("wrist", start_connected=True),
            front=_Camera("front", fail_times=99),
        )
        hw = _robot(follower)

        with caplog.at_level("ERROR"):
            _connect(hw)

        messages = [record.getMessage() for record in caplog.records]
        stale_logs = [text for text in messages if "not fully connected" in text.lower()]
        assert stale_logs, messages
        assert "wrist" in stale_logs[0]
        assert stale_logs[0].isascii()


class TestRollbackIsResilient:
    def test_one_undisconnectable_camera_does_not_block_the_others(self):
        """Best-effort per camera: a raising disconnect must not abort the sweep."""
        follower = _Follower(
            a=_UndisconnectableCamera("a", start_connected=True),
            b=_Camera("b", start_connected=True),
            c=_Camera("c", fail_times=99),
        )
        hw = _robot(follower)

        ok, _ = _connect(hw)

        assert ok is False
        assert follower.cameras["b"].disconnect_calls == 1, "the sweep stopped at the bad camera"
        assert not follower.cameras["b"].is_connected

    def test_a_robot_without_cameras_is_unaffected(self):
        """The bus-only rollback contract must be byte-identical."""

        class _NoCameras:
            def __init__(self) -> None:
                self.bus = _Bus()

            def connect(self, calibrate: bool = True) -> None:
                self.bus.connect()
                raise ConnectionError("handshake failed")

            @property
            def is_connected(self) -> bool:
                return False

            @property
            def is_calibrated(self) -> bool:
                return True

        follower = _NoCameras()
        hw = _robot(follower)

        ok, _ = _connect(hw)

        assert ok is False
        assert not follower.bus.is_connected
        assert follower.bus.disconnect_calls == 1

    def test_rollback_is_callable_directly_and_idempotent(self):
        follower = _Follower(wrist=_Camera("wrist", start_connected=True))
        follower.bus.connect()
        hw = _robot(follower)

        hw._rollback_half_open_connect()
        hw._rollback_half_open_connect()

        assert _fully_closed(follower)
        # Second sweep must not re-disconnect an already-closed camera.
        assert follower.cameras["wrist"].disconnect_calls == 1
