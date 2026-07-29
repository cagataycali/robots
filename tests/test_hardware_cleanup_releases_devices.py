"""``cleanup()`` must release the devices it documents, not just library state.

``Robot.cleanup()`` tears down everything the *library* owns -- the shutdown
latch, the task executor, the mesh client, the ROS bridge -- and both API
tables document it as "stop tasks, disconnect the arm and cameras, stop mesh".
The one resource that is a physical device was the only thing it left alone:
the ``FeetechMotorsBus`` serial port, plus one ``/dev/video*`` node and one
read thread per camera.

That is not tidiness. A serial port is exclusive on Linux and macOS, so a
second process -- or a re-constructed ``Robot`` in this one -- cannot open the
same ``/dev/tty*`` afterwards, which is exactly the documented recovery for a
wedged arm. ``cleanup()`` is also what ``__del__`` calls, so a script that
finishes normally left the arm energised at its last commanded position instead
of going through the driver's own disconnect, where the Feetech
``disable_torque_on_disconnect`` write lives. And nothing else could recover
it: the executor is shut down and the shutdown is latched, so no library entry
point remained that would reach a disconnect.

``stop()`` -- the async spelling of the same teardown -- had the mirror-image
defect. It called ``self.robot.disconnect()`` *before* ``cleanup()``, and
lerobot's ``Robot.disconnect()`` is ``@check_if_not_connected``, so stopping a
robot that was never connected raised ``DeviceNotConnectedError`` inside
``stop()``'s own handler and ``cleanup()`` was never reached at all: the
executor kept running and the shutdown stayed unlatched, so the terminal
guarantee ``stop()`` documents did not hold.

These tests pin the whole teardown contract:

    - a connected robot's bus and cameras are closed, exactly once each, and
      through the driver's own ``disconnect()`` so the torque write happens;
    - the disconnect runs *after* the task executor has drained, so a rollout
      still finishing cannot command a port being closed underneath it;
    - a half-open device set -- which the driver's own ``disconnect()`` refuses
      to touch because it is gated on every device -- is still closed;
    - one device whose close raises does not keep the rest of the set open, and
      does not propagate out of the teardown;
    - ``cleanup()`` is idempotent, and safe on a driver that exposes no devices
      at all;
    - ``stop()`` is terminal whether or not the robot was ever connected.

No serial port and no camera device is opened: the lerobot driver and its
cameras are the in-memory fakes from ``test_hardware_camera_rollback``, which
mirror lerobot's connect ordering, its unguarded disconnect loop and its
``is_connected`` / decorator contracts.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from tests.test_hardware_camera_rollback import (
    _FakeCamera,
    _FakeCameraRobot,
    _make_robot,
)

DEADLINE = 5.0
"""Seconds. A blocked teardown must fail the test, not hang the suite."""


def _connected_robot(**camera_kwargs: Any) -> tuple[Any, Any]:
    """A one-camera arm, connected, plus the ``Robot`` wrapping it."""
    fake = _FakeCameraRobot({"wrist": _FakeCamera("wrist_cam", **camera_kwargs)})
    hw = _make_robot(fake)
    ok, err = asyncio.run(hw._connect_robot())
    assert ok is True, err
    assert fake.is_connected is True
    return fake, hw


class TestCleanupReleasesTheDevices:
    """The physical devices are part of what ``cleanup()`` releases."""

    def test_the_bus_and_every_camera_are_closed(self) -> None:
        """Leaving them open holds the serial port for the life of the
        process, so the documented tear-down-and-reconnect recovery cannot
        work without exiting."""
        fake, hw = _connected_robot()

        hw.cleanup()

        assert fake.bus.is_connected is False
        assert fake.cameras["wrist"].is_connected is False
        assert fake.is_connected is False

    def test_the_driver_disconnect_does_the_torque_write(self) -> None:
        """Going through the driver's own ``disconnect()`` is what de-energises
        the arm: ``disable_torque_on_disconnect`` lives there, so force-closing
        the port instead would leave the arm holding its last command."""
        fake, hw = _connected_robot()

        hw.cleanup()

        assert fake.bus.disconnect_calls == [True]
        assert fake.cameras["wrist"].disconnect_calls == 1

    def test_a_second_cleanup_does_not_disconnect_again(self) -> None:
        """``cleanup()`` is reachable twice -- explicitly and then from
        ``__del__`` -- and a device that is already closed raises
        ``DeviceNotConnectedError`` if asked again."""
        fake, hw = _connected_robot()

        hw.cleanup()
        hw.cleanup()

        assert fake.bus.disconnect_calls == [True]
        assert fake.cameras["wrist"].disconnect_calls == 1

    def test_a_driver_exposing_no_devices_is_tolerated(self) -> None:
        """``robot=`` accepts any lerobot ``Robot``, and the base class
        declares ``disconnect()`` as a no-op with no bus or camera attributes
        at all. Teardown must not depend on them existing."""

        class _Minimal:
            is_connected = False

            def disconnect(self) -> None:
                raise AssertionError("must not be called on a disconnected robot")

        hw = _make_robot(_Minimal())

        hw.cleanup()  # must not raise

        assert hw._shutdown_event.is_set() is True


class TestCleanupOrdering:
    """The port must not close under a rollout that is still draining."""

    def test_the_disconnect_waits_for_the_task_executor(self) -> None:
        """``send_action`` on a closed port is the failure this ordering
        exists to prevent, so the disconnect has to follow
        ``executor.shutdown(wait=True)`` rather than precede it."""
        fake, hw = _connected_robot()
        log: list[str] = []
        gate = threading.Event()

        def _draining_rollout() -> None:
            log.append("rollout-start")
            gate.wait(DEADLINE)
            log.append(f"rollout-sees-bus-open={fake.bus.is_connected}")
            log.append("rollout-end")

        original_disconnect = fake.bus.disconnect

        def _recording_disconnect(disable_torque: bool = True) -> None:
            log.append("bus-disconnect")
            original_disconnect(disable_torque)

        fake.bus.disconnect = _recording_disconnect  # type: ignore[method-assign]

        hw._executor.submit(_draining_rollout)
        # Opened from a third thread, so the rollout genuinely spans the
        # teardown instead of finishing before it starts.
        threading.Timer(0.2, gate.set).start()

        hw.cleanup()

        assert log == [
            "rollout-start",
            "rollout-sees-bus-open=True",
            "rollout-end",
            "bus-disconnect",
        ]


class TestCleanupIsBestEffort:
    """One device that cannot be closed must not keep the others open."""

    def test_a_half_open_set_is_still_closed(self) -> None:
        """``Robot.disconnect()`` is gated on ``is_connected`` -- the bus *and*
        every camera -- so it refuses to run while any one camera is shut. The
        port would otherwise stay open with no library path left to release
        it."""
        fake, hw = _connected_robot()
        fake.cameras["wrist"].is_connected = False  # e.g. a camera that dropped out
        assert fake.is_connected is False
        assert fake.bus.is_connected is True

        hw.cleanup()

        assert fake.bus.is_connected is False
        # Force-closed rather than through the driver: its ordered disconnect,
        # and with it the torque write, could not run on a half-open set.
        assert fake.bus.disconnect_calls == [False]

    def test_a_camera_that_cannot_close_does_not_keep_the_rest_open(self) -> None:
        """lerobot's ``disconnect()`` is one unguarded loop, so the first close
        that raises abandons every device after it. Closing each device
        independently afterwards is what bounds the damage to the stuck one."""
        fake = _FakeCameraRobot(
            {
                "stuck": _FakeCamera("stuck_cam", close_raises=True),
                "healthy": _FakeCamera("healthy_cam"),
            }
        )
        hw = _make_robot(fake)
        asyncio.run(hw._connect_robot())
        assert fake.is_connected is True

        hw.cleanup()  # must not raise

        assert fake.bus.is_connected is False
        assert fake.cameras["healthy"].is_connected is False
        assert fake.cameras["stuck"].is_connected is True

    def test_a_driver_disconnect_that_raises_does_not_abort_the_teardown(self) -> None:
        """``cleanup()`` releases every remaining resource whatever else has
        failed, so a driver that raises on disconnect must not stop the mesh
        and ROS bridges from being torn down."""
        fake, hw = _connected_robot()
        mesh_stops: list[str] = []
        hw.mesh = type("Mesh", (), {"stop": lambda _self: mesh_stops.append("stopped")})()

        def _raising_disconnect() -> None:
            raise OSError("bus went away")

        fake.disconnect = _raising_disconnect  # type: ignore[method-assign]

        hw.cleanup()  # must not raise

        assert mesh_stops == ["stopped"]
        # Swept up device by device once the driver's own loop could not run.
        assert fake.bus.is_connected is False
        assert fake.cameras["wrist"].is_connected is False


class TestStopIsTerminal:
    """``stop()`` is the async spelling of ``cleanup()``, on every path."""

    def test_stopping_a_robot_that_was_never_connected_still_shuts_it_down(self) -> None:
        """``Robot.disconnect()`` raises ``DeviceNotConnectedError`` when the
        robot is not connected. Doing it ahead of ``cleanup()`` meant that
        error ended the teardown early, leaving the executor running and the
        shutdown unlatched -- so the terminal guarantee did not hold for the
        most ordinary case of all."""
        fake = _FakeCameraRobot({"wrist": _FakeCamera("wrist_cam")})
        hw = _make_robot(fake)
        executor = hw._executor
        assert fake.is_connected is False

        asyncio.run(asyncio.wait_for(hw.stop(), timeout=DEADLINE))

        assert hw._shutdown_event.is_set() is True
        assert executor._shutdown is True
        assert fake.bus.disconnect_calls == []

    def test_stopping_a_connected_robot_disconnects_it_once(self) -> None:
        """The disconnect moved into ``cleanup()``, so it must not also happen
        in ``stop()``: a second one would raise on the now-closed devices."""
        fake, hw = _connected_robot()
        executor = hw._executor

        asyncio.run(asyncio.wait_for(hw.stop(), timeout=DEADLINE))

        assert fake.bus.disconnect_calls == [True]
        assert fake.cameras["wrist"].disconnect_calls == 1
        assert fake.is_connected is False
        assert executor._shutdown is True
