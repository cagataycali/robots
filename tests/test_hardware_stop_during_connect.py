# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A stop pressed while the robot is still connecting must actually stop it.

``stop_task`` early-returned ``{"status": "success", "No task running to stop"}``
for every state that was not ``RUNNING``. But ``_execute_task_async`` sits in
``CONNECTING`` for the whole hardware bring-up - a FeetechMotorsBus handshake plus
``warmup_s`` per camera, 2-3 s on a real SO-101 and longer on a multi-camera rig.

Measured against the real loop with a 2.5 s ``connect()``:

    stop at t=0.9s  status=connecting -> 'No task running to stop (current: connecting)'
    stop at t=1.2s  status=connecting -> 'No task running to stop (current: connecting)'
    stop at t=2.0s  status=connecting -> 'No task running to stop (current: connecting)'
    servo writes after the stops: 104
    final status: completed

Three stop presses reported success, then the arm moved anyway. ``mesh/core.py``
routes the fleet ``{"action": "stop"}`` straight into ``stop_task``, so the mesh
e-stop inherited the same hole.

``stop_task`` now sets a ``_stop_requested`` latch unconditionally and BEFORE the
status check, the guard accepts ``CONNECTING``, and the latch is checked right
after connect (before the policy is even built) and in both loop conditions.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class _SlowConnectRobot:
    """A driver whose connect() blocks, like a real motor-bus + camera warmup."""

    def __init__(self, connect_seconds: float = 1.0) -> None:
        self._connect_seconds = connect_seconds
        self._connected = False
        self.sent: list[float] = []
        self.disconnected = False

    def connect(self, calibrate: bool = False) -> None:
        time.sleep(self._connect_seconds)
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        return dict.fromkeys(_KEYS, 0.0)

    def send_action(self, action: dict[str, float]) -> None:
        self.sent.append(time.perf_counter())

    def disconnect(self) -> None:
        self.disconnected = True


class _ChunkPolicy(Policy):
    @property
    def provider_name(self) -> str:
        return "fake"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(_KEYS, 0.0) for _ in range(8)]


def _robot(driver: _SlowConnectRobot) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = 50.0
    hw.action_sleep_time = 0.02
    hw.action_horizon = 8
    hw.robot = driver
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_state = hardware_robot.RobotTaskState()
    hw.mesh = None
    hw.peer_id = None
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm")
    hw._publish_ros_telemetry = lambda observation: None  # type: ignore[assignment,method-assign,misc]
    return hw


class TestStopDuringConnect:
    def test_stop_while_connecting_prevents_all_motion(self):
        """The regression: 3 ignored stops then 104 servo writes."""
        driver = _SlowConnectRobot(connect_seconds=1.0)
        hw = _robot(driver)

        async def scenario():
            task = asyncio.create_task(hw._execute_task_async("t", policy_object=_ChunkPolicy(), duration=2.0))
            await asyncio.sleep(0.3)  # firmly inside the CONNECTING window
            assert hw._task_state.status is TaskStatus.CONNECTING
            result = hw.stop_task()
            await task
            return result

        result = asyncio.run(scenario())

        # The stop must not claim there was nothing to stop ...
        assert "No task running" not in result["content"][0]["text"], result
        # ... the arm must never have moved ...
        assert driver.sent == [], f"{len(driver.sent)} servo writes after a stop"
        # ... and the task must end STOPPED, not COMPLETED.
        assert hw._task_state.status is TaskStatus.STOPPED

    def test_stop_is_latched_even_before_the_status_is_set(self):
        """A stop must never be lost to a state the guard does not recognise."""
        hw = _robot(_SlowConnectRobot())

        hw.stop_task()  # status is IDLE here

        assert hw._stop_requested.is_set()

    def test_latch_is_cleared_so_the_next_rollout_runs(self):
        """A stop must not permanently poison the robot."""
        driver = _SlowConnectRobot(connect_seconds=0.0)
        hw = _robot(driver)
        hw.stop_task()  # latch a stop while idle
        assert hw._stop_requested.is_set()

        asyncio.run(hw._execute_task_async("t", policy_object=_ChunkPolicy(), n_steps=3, duration=5.0))

        assert driver.sent, "a stale latch blocked the next rollout"
        assert hw._task_state.status is TaskStatus.COMPLETED


class TestStopDuringRun:
    def test_stop_mid_rollout_still_works(self):
        """The pre-existing RUNNING path must keep working."""
        driver = _SlowConnectRobot(connect_seconds=0.0)
        hw = _robot(driver)

        async def scenario():
            task = asyncio.create_task(hw._execute_task_async("t", policy_object=_ChunkPolicy(), duration=5.0))
            await asyncio.sleep(0.2)
            sent_at_stop = len(driver.sent)
            hw.stop_task()
            await task
            return sent_at_stop

        sent_at_stop = asyncio.run(scenario())

        assert hw._task_state.status is TaskStatus.STOPPED
        # At most one more action lands (the one already in flight).
        assert len(driver.sent) - sent_at_stop <= 1


class TestIdleStopUnchanged:
    def test_stop_when_idle_reports_no_task(self):
        """The documented idle behaviour is preserved for genuinely idle states."""
        hw = _robot(_SlowConnectRobot())

        result = hw.stop_task()

        assert result["status"] == "success"
        assert "No task running to stop" in result["content"][0]["text"]

    @pytest.mark.parametrize("state", [TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.STOPPED])
    def test_terminal_states_report_no_task(self, state):
        hw = _robot(_SlowConnectRobot())
        hw._task_state.status = state

        assert "No task running to stop" in hw.stop_task()["content"][0]["text"]


class TestStopOnAHalfBuiltRobot:
    def test_stop_works_without_init_having_run(self):
        """A stop is the one call that must work on a partially built Robot.

        ``__init__`` can raise partway through (``_initialize_robot`` touching
        hardware), and that is exactly when an operator reaches for the stop.
        """
        hw = Robot.__new__(Robot)
        hw.tool_name_str = "half_built"
        hw._task_state = hardware_robot.RobotTaskState()

        result = hw.stop_task()

        assert result["status"] == "success"
        assert hw._stop_requested.is_set()
