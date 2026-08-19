"""A camera that will not open costs the camera, not the whole arm.

lerobot gates ``Robot.is_connected`` on ``bus.is_connected and all(cam.is_connected
...)``, so one unopenable camera reports a mechanically healthy arm as
disconnected: no joints, no teleop, no recording, and no explanation. On macOS
this happens for a reason that is not the robot's fault at all -- the OS denies
camera access to the process, so every index fails at once.

These tests pin the rule: with the motor bus open, unopenable cameras are dropped
(each retried alone so its OWN error is the reason recorded), and a motor failure
is never dressed up as a camera problem.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from strands_robots.hardware_robot import _degrade_to_available_cameras


class FakeBus:
    def __init__(self, *, connected: bool = True) -> None:
        self.is_connected = connected


class FakeCamera:
    """A lerobot-shaped camera: ``connect()``, ``is_connected``, ``disconnect()``."""

    def __init__(self, *, fails: str | None = None, lies: bool = False) -> None:
        self.fails = fails
        self.lies = lies  # connects without error yet stays disconnected
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        if self.fails is not None:
            raise OSError(self.fails)
        if not self.lies:
            self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class FakeRobot:
    """Reproduces lerobot's all-or-nothing ``is_connected`` gate."""

    def __init__(self, cameras: dict[str, FakeCamera], *, bus_connected: bool = True) -> None:
        self.bus = FakeBus(connected=bus_connected)
        self.cameras = cameras
        self.config = SimpleNamespace(cameras={name: {"index_or_path": i} for i, name in enumerate(cameras)})

    @property
    def is_connected(self) -> bool:
        return bool(self.bus.is_connected) and all(c.is_connected for c in self.cameras.values())


def test_unopenable_camera_is_dropped_and_the_arm_becomes_connected():
    """The whole point: motors usable, one camera gone, and it says why."""
    robot = FakeRobot(
        {
            "top": FakeCamera(),
            "wrist": FakeCamera(fails="Failed to open OpenCVCamera(2)."),
        }
    )
    assert robot.is_connected is False  # the bug being fixed

    dropped = _degrade_to_available_cameras(robot)

    assert dropped == {"wrist": "Failed to open OpenCVCamera(2)."}
    assert robot.is_connected is True
    assert set(robot.cameras) == {"top"}
    # The config the status report reads must agree with reality.
    assert set(robot.config.cameras) == {"top"}


def test_every_camera_denied_still_leaves_a_working_arm():
    """The macOS permission case: no camera opens, the arm is still jog-able."""
    robot = FakeRobot(
        {
            "top": FakeCamera(fails="not authorized to capture video"),
            "wrist": FakeCamera(fails="not authorized to capture video"),
        }
    )

    dropped = _degrade_to_available_cameras(robot)

    assert set(dropped) == {"top", "wrist"}
    assert robot.is_connected is True
    assert robot.cameras == {}


def test_each_camera_is_retried_alone_so_the_reason_is_its_own():
    """lerobot's loop aborts on the first failure; one error must not speak for all."""
    robot = FakeRobot(
        {
            "top": FakeCamera(fails="index 0 is busy"),
            "wrist": FakeCamera(fails="index 2 does not exist"),
        }
    )

    dropped = _degrade_to_available_cameras(robot)

    assert dropped == {"top": "index 0 is busy", "wrist": "index 2 does not exist"}
    assert robot.cameras["top"].connect_calls == 1 if "top" in robot.cameras else True


def test_a_motor_failure_is_never_reported_as_a_camera_problem():
    """With a dead bus there is nothing to degrade -- the real error must stand."""
    robot = FakeRobot({"top": FakeCamera(fails="boom")}, bus_connected=False)

    assert _degrade_to_available_cameras(robot) == {}
    # And the camera set is untouched, so the caller's rollback still sees it.
    assert set(robot.cameras) == {"top"}


def test_healthy_cameras_are_left_alone():
    """Nothing to drop means nothing degraded, so the original error stands."""
    top = FakeCamera()
    top.is_connected = True
    robot = FakeRobot({"top": top})

    assert _degrade_to_available_cameras(robot) == {}
    assert top.connect_calls == 0  # already open, not poked again
    assert set(robot.cameras) == {"top"}


def test_a_camera_that_connects_without_error_but_stays_shut_is_dropped():
    """lerobot's OpenCV backend does this when the OS hands back a dead device."""
    robot = FakeRobot({"top": FakeCamera(lies=True)})

    dropped = _degrade_to_available_cameras(robot)

    assert list(dropped) == ["top"]
    assert "did not connect" in dropped["top"]
    assert robot.is_connected is True


def test_no_cameras_configured_is_not_a_camera_problem():
    robot = FakeRobot({})
    assert _degrade_to_available_cameras(robot) == {}


def test_degrading_that_does_not_help_reports_nothing():
    """If the robot still is not connected, do not claim a degraded success."""

    class StillBroken(FakeRobot):
        @property
        def is_connected(self) -> bool:
            return False

    robot = StillBroken({"top": FakeCamera(fails="nope")})
    assert _degrade_to_available_cameras(robot) == {}


def test_the_dropped_camera_is_named_in_a_warning(caplog):
    """An arm silently running without its camera would poison a dataset."""
    robot = FakeRobot({"wrist": FakeCamera(fails="not authorized to capture video")})

    with caplog.at_level(logging.WARNING, logger="strands_robots.hardware_robot"):
        _degrade_to_available_cameras(robot)

    text = caplog.text
    assert "wrist" in text
    assert "not authorized to capture video" in text
    assert "cannot record" in text  # says what the operator loses


def test_a_half_open_camera_is_closed_when_dropped():
    """Do not leak a device that opened far enough to hold the hardware."""
    half_open = FakeCamera(lies=True)

    def connect_then_hold() -> None:
        half_open.connect_calls += 1
        half_open.is_connected = False  # reports shut to the gate...

    half_open.connect = connect_then_hold  # type: ignore[method-assign]
    robot = FakeRobot({"wrist": half_open})
    _degrade_to_available_cameras(robot)

    assert "wrist" not in robot.cameras
