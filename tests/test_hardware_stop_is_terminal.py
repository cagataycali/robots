# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``stop()`` must be terminal whatever state the robot is in.

``Robot.stop()`` is documented as the async spelling of :meth:`cleanup` -- the
docs say "``cleanup()`` (and ``stop()``, which calls it) is terminal" -- but it
disconnected the driver itself *before* delegating, and did so unguarded::

    if hasattr(self.robot, "disconnect"):
        await asyncio.to_thread(self.robot.disconnect)
    self.cleanup()

lerobot gates ``Robot.disconnect()`` on ``is_connected``
(``bus.is_connected and all(cam.is_connected ...)``) and raises
``DeviceNotConnectedError`` when it is false, so that call raised for any robot
that was not *fully* connected -- one that never connected at all, or one a
failed bring-up left disconnected. ``stop()`` swallows exceptions to stay
fail-soft for an operator, so the raise was logged and ``cleanup()`` was never
reached. Measured with a driver double, one call to ``stop()``:

    robot state          shutdown latched   executor running   devices closed
    fully connected            yes                no                yes
    never connected            NO                 YES               n/a

So the most ordinary case -- construct a ``Robot``, stop it without having run
a task, or stop it twice -- silently kept every terminal guarantee unmet: no
shutdown latch, a task executor still accepting work, and any device a
half-open connect had opened still held, with no entry point left that would
close it.

Ordering was wrong too, in a way that only became visible once ``cleanup()``
learned to close the devices. ``cleanup()`` closes them *last*, after the
teleop loop, the task executor, the mesh and the ROS bridge are down, because
``send_action`` re-opens the robot lazily on a command that finds it
disconnected; disconnecting in ``stop()`` put that close ahead of every one of
those command sources. And ``cleanup()`` was awaited inline from an ``async
def`` while it joins the task executor and closes a serial port, so it blocked
the event loop for the whole drain.

What these tests pin:

    - a robot that was never connected, and one left disconnected by a failed
      bring-up, are still latched shut with the executor released;
    - a device that a failed close left open is closed;
    - the rollout entry points refuse afterwards, naming the shutdown;
    - the connected path is unchanged -- the driver's own ``disconnect()`` is
      still what runs, because that is where torque disable lives -- and a
      disconnect that raises no longer aborts the rest of the teardown;
    - the devices close after the executor drains and after the mesh stops;
    - the event loop keeps running while ``stop()`` drains;
    - a rollout still in bring-up when ``stop()`` lands reports ``STOPPED``,
      not ``COMPLETED``.

No serial port, camera node or arm is involved: the lerobot driver and its
cameras are in-memory fakes mirroring lerobot's connect ordering and its
``is_connected`` / decorator contracts, and each bring-up stage is an event the
test opens explicitly.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import TaskStatus
from strands_robots.policies import Policy
from tests.test_hardware_camera_rollback import _FakeCamera, _FakeCameraRobot, _make_robot

# Generous ceiling for every bounded wait: each is satisfied by an event the
# test itself sets, so reaching it means the contract is broken, not that the
# host was slow.
DEADLINE = 10.0


def _executor_accepts_work(hw: HwRobot) -> bool:
    """Whether the task executor would still run a job.

    Behavioural rather than a private-flag read: a shut-down
    ``ThreadPoolExecutor`` raises ``RuntimeError`` on submit. The probe job is
    cancelled so a live executor is left with no extra worker.
    """
    try:
        hw._executor.submit(lambda: None).cancel()
    except RuntimeError:
        return False
    return True


def _one_camera_arm(**camera_kwargs: Any) -> _FakeCameraRobot:
    return _FakeCameraRobot({"wrist": _FakeCamera("wrist_cam", **camera_kwargs)})


class TestStopIsTerminalWhateverTheConnectionState:
    """The guarantee cannot depend on the robot being fully connected."""

    def test_a_robot_that_was_never_connected_is_still_shut_down(self) -> None:
        """The most ordinary case: construct a ``Robot``, stop it without ever
        running a task. The driver refuses the disconnect, and that refusal
        must not cost the shutdown latch or the executor release."""
        hw = _make_robot(_one_camera_arm())

        asyncio.run(hw.stop())

        assert hw._shutdown_event.is_set()
        assert _executor_accepts_work(hw) is False

    def test_a_robot_left_disconnected_by_a_failed_bring_up_is_still_shut_down(self) -> None:
        """A failed connect rolls its devices back, so the robot reports
        ``is_connected`` False -- exactly the state the driver's disconnect
        refuses. Stopping such a robot is the documented recovery, and it has
        to reach the teardown."""
        fake = _FakeCameraRobot({"wrist": _FakeCamera("wrist_cam"), "top": _FakeCamera("top_cam", fails=True)})
        hw = _make_robot(fake)

        ok, _ = asyncio.run(hw._connect_robot())
        assert ok is False
        assert fake.is_connected is False

        asyncio.run(hw.stop())

        assert hw._shutdown_event.is_set()
        assert _executor_accepts_work(hw) is False

    def test_a_device_a_failed_close_left_open_is_closed(self) -> None:
        """The bring-up rollback is best-effort per device, so a close that
        raises leaves that camera streaming. ``stop()`` is the last chance to
        get the node released, and it only gets one if it reaches the
        teardown."""

        class _StickyOnce(_FakeCamera):
            """Refuses its first close, accepts the next -- a transient fault."""

            def disconnect(self) -> None:
                if self.disconnect_calls == 0:
                    self.disconnect_calls += 1
                    raise OSError(f"{self.name} close failed")
                super().disconnect()

        wrist = _StickyOnce("wrist_cam")
        fake = _FakeCameraRobot({"wrist": wrist, "top": _FakeCamera("top_cam", fails=True)})
        hw = _make_robot(fake)

        asyncio.run(hw._connect_robot())
        assert wrist.is_connected is True, "precondition: the failed close left it open"

        asyncio.run(hw.stop())

        assert wrist.is_connected is False

    def test_the_rollout_entry_points_refuse_afterwards(self) -> None:
        """``stop()`` is terminal, so it must latch the same shutdown
        ``cleanup()`` does -- otherwise a rollout is admitted that re-opens the
        hardware and commands the arm zero times."""
        hw = _make_robot(_one_camera_arm())

        asyncio.run(hw.stop())

        for result in (
            hw.run_policy(_SilentPolicy(), "pick the cube", duration=1.0),
            hw.start_task("pick the cube", policy_port=5555),
        ):
            assert result["status"] == "error"
            assert "shut down" in result["content"][0]["text"].lower()

    def test_stopping_a_never_connected_robot_does_not_raise(self) -> None:
        """Fail-soft is the reason the defect was silent, and it is still the
        contract: an operator teardown must not propagate a device fault."""
        hw = _make_robot(_one_camera_arm())

        asyncio.run(hw.stop())  # must not raise
        asyncio.run(hw.stop())  # idempotent


class TestTheConnectedPathIsUnchanged:
    """Delegating must not lose what the disconnect was there for."""

    def test_the_drivers_own_disconnect_still_runs(self) -> None:
        """Torque disable and gripper release live in the driver's
        ``disconnect()``, so it stays preferred while it is callable -- closing
        the port underneath it would leave the arm energised."""
        fake = _one_camera_arm()
        fake.connect()
        hw = _make_robot(fake)

        asyncio.run(hw.stop())

        assert fake.bus.disconnect_calls == [True], "torque disable requested"
        assert fake.cameras["wrist"].is_connected is False
        assert hw._shutdown_event.is_set()

    def test_a_disconnect_that_raises_does_not_abort_the_teardown(self) -> None:
        """The USB cable pulled mid-session: the driver's disconnect raises.
        That is one best-effort step, and it must not take the shutdown latch,
        the executor release or the remaining devices with it."""

        class _RaisingDisconnect(_FakeCameraRobot):
            def disconnect(self) -> None:
                raise RuntimeError("device already gone")

        fake = _RaisingDisconnect({"wrist": _FakeCamera("wrist_cam")})
        fake.connect()
        hw = _make_robot(fake)

        asyncio.run(hw.stop())  # must not raise

        assert hw._shutdown_event.is_set()
        assert _executor_accepts_work(hw) is False
        # The per-device fallback still gets the port and the camera shut.
        assert fake.bus.is_connected is False
        assert fake.cameras["wrist"].is_connected is False


class TestTheTeardownKeepsCleanupsOrdering:
    """``send_action`` re-opens the robot lazily, so the close must come last."""

    def test_the_devices_close_after_the_executor_drains(self) -> None:
        """A job still draining can command the arm, and a command that finds
        the robot disconnected re-opens it -- behind a teardown that has no
        step left to close it again."""
        log: list[str] = []
        gate = threading.Event()

        class _LoggingBusRobot(_FakeCameraRobot):
            def disconnect(self) -> None:
                log.append("devices_closed")
                super().disconnect()

        fake = _LoggingBusRobot({})
        fake.connect()
        hw = _make_robot(fake)

        def slow_job() -> None:
            gate.wait(DEADLINE)
            log.append("executor_job_finished")

        hw._executor.submit(slow_job)
        threading.Timer(0.2, gate.set).start()

        asyncio.run(hw.stop())

        assert log == ["executor_job_finished", "devices_closed"]

    def test_the_mesh_stops_before_the_devices_close(self) -> None:
        """The mesh input receiver applies frames through ``send_action``, so
        it is a live command source until its ``stop()`` returns."""
        log: list[str] = []

        class _LoggingRobot(_FakeCameraRobot):
            def disconnect(self) -> None:
                log.append("devices_closed")
                super().disconnect()

        class _Mesh:
            def stop(self) -> None:
                log.append("mesh_stopped")

        fake = _LoggingRobot({})
        fake.connect()
        hw = _make_robot(fake)
        hw.mesh = _Mesh()

        asyncio.run(hw.stop())

        assert log == ["mesh_stopped", "devices_closed"]


class TestStopDoesNotBlockTheEventLoop:
    """``cleanup()`` blocks; ``stop()`` is ``async`` and must not."""

    def test_other_tasks_keep_running_while_stop_drains(self) -> None:
        """``cleanup()`` joins the task executor and closes a serial port. Run
        inline, that stalls every other coroutine on the loop -- a mesh
        session, a ROS spin, a supervising agent -- for the whole drain."""
        fake = _one_camera_arm()
        fake.connect()
        hw = _make_robot(fake)
        gate = threading.Event()
        hw._executor.submit(gate.wait, DEADLINE)

        async def drive() -> int:
            ticks = 0

            async def heartbeat() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            beat = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.02)
            threading.Timer(0.3, gate.set).start()
            before = ticks
            await hw.stop()
            during = ticks - before
            beat.cancel()
            try:
                await beat
            except asyncio.CancelledError:  # expected: task was explicitly cancelled above
                pass
            return during

        # A 10 ms heartbeat across a ~0.3 s drain gives ~30 ticks; blocking the
        # loop gives exactly none. The bound is deliberately far below that so a
        # loaded host cannot fail it.
        assert asyncio.run(drive()) >= 3


class _SilentPolicy(Policy):
    """Commands nothing: these tests are about teardown, not motion."""

    @property
    def provider_name(self) -> str:
        return "test"

    def set_robot_state_keys(self, keys: list[str]) -> None:
        return None

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.0}]


class _BringUpRig:
    """A ``Robot`` whose bring-up stages are gates the test opens explicitly.

    The driver is a lerobot-faithful double, so its ``disconnect()`` refuses
    while the robot is not connected -- which is what made a ``stop()`` during
    bring-up a no-op.
    """

    def __init__(self) -> None:
        self.arm = _one_camera_arm()
        self.connect_gate = threading.Event()
        self.result: dict[str, Any] | None = None

        hw = _make_robot(self.arm)
        hw.control_frequency = 500.0
        hw.action_sleep_time = 1.0 / 500.0
        hw.action_horizon = 1

        async def connect() -> tuple[bool, str]:
            await asyncio.to_thread(self.connect_gate.wait, DEADLINE)
            return True, ""

        async def initialize_policy(policy: Any) -> bool:  # noqa: ARG001 - stage stub
            return True

        def publish(observation: dict[str, Any], skip_images: bool = False) -> None:  # noqa: ARG001
            return None

        hw._connect_robot = connect  # type: ignore[method-assign]
        hw._initialize_policy = initialize_policy  # type: ignore[method-assign]
        hw._publish_ros_telemetry = publish  # type: ignore[method-assign]
        self.robot = hw

    def start(self) -> threading.Thread:
        def target() -> None:
            self.result = self.robot.run_policy(_SilentPolicy(), "pick the cube", duration=5.0, n_steps=3)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def wait_for_connecting(self) -> None:
        deadline = time.monotonic() + DEADLINE
        while time.monotonic() < deadline:
            if self.robot._task_state.status == TaskStatus.CONNECTING:
                return
            time.sleep(0.005)
        pytest.fail(f"never reached CONNECTING (status={self.robot._task_state.status.value})")


class TestAStopDuringBringUpTruncatesTheRollout:
    """The bring-up window is seconds on a real arm, so it is where a stop lands."""

    def test_a_rollout_still_connecting_is_reported_stopped(self) -> None:
        """``cleanup()`` latches the shutdown before it reads the task status,
        and the rollout's terminal decision consults that latch -- so a task
        interrupted in bring-up reports a truncation. Skipping ``cleanup()``
        left the latch clear, and the rollout then ran on and reported itself
        ``COMPLETED``."""
        rig = _BringUpRig()
        thread = rig.start()
        rig.wait_for_connecting()

        asyncio.run(rig.robot.stop())
        rig.connect_gate.set()
        thread.join(timeout=DEADLINE)

        assert not thread.is_alive(), "rollout thread did not finish"
        assert rig.robot._task_state.status == TaskStatus.STOPPED
        assert rig.arm.sent_actions == []
