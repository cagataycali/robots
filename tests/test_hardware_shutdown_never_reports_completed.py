# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A shut-down robot must refuse task work, not report a rollout it never drove.

``cleanup`` (and ``stop``, which calls it) sets ``_shutdown_event`` and never
clears it, then shuts the task executor down and tears down the mesh and ROS
bridges. ``_execute_task_async``'s loop condition honors that latch - but it was
the only thing that did, so the latch produced two different misreports.

A task started after ``cleanup`` was admitted, took the motors-bus claim,
commanded the arm zero times, and was reported as finished work::

    run 1 status: success | commands: 176
    cleanup()
    run 2 status: success
    run 2 text  : Policy rollout completed: 0 steps in 0.0s
    run 2 commands: 0

``start_task`` was worse still: its submit reached the already-shut-down
executor and surfaced a bare ``RuntimeError: cannot schedule new futures after
shutdown`` past a method whose contract is a tool-shaped result.

The second misreport is the more serious one, because it can strike a rollout
that really was driving the arm. ``cleanup`` sets the latch FIRST and only then
calls ``stop_task`` for a rollout it finds ``RUNNING``; a loop that has already
exited on the latch by that point is no longer ``RUNNING``, so ``stop_task`` is
never called and the stop latch stays clear. The terminal block consulted only
the stop latch, so a rollout truncated two hundredths of a second into a
thirty-second budget reported itself complete::

    status      : success
    text        : Policy rollout completed: 24 steps in 0.0s
    task status : TaskStatus.COMPLETED
    stop latch  : False | shutdown latch: True

These tests pin both halves of the contract: the entry points refuse a shut-down
robot by name, and the terminal block treats a shutdown as the interruption it
is. A rollout that reaches its own budget is still ``COMPLETED``, so the guard
cannot swallow a genuine success.

No serial port is opened and no arm is commanded: the lerobot driver is an
in-memory fake and every latch transition is one the test performs explicitly,
so nothing here depends on wall-clock timing.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy

_JOINT = "gripper.pos"


class _FakeArm:
    """In-memory stand-in for a connected lerobot robot.

    ``on_step`` fires after the Nth command reaches the bus, which is how a
    latch transition is placed at an exact point in the rollout instead of
    after a sleep.
    """

    def __init__(self, on_step: tuple[int, Any] | None = None) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.sent_actions: list[dict[str, Any]] = []
        self._on_step = on_step

    def connect(self, calibrate: bool = False) -> None:
        return None

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_observation(self) -> dict[str, Any]:
        return {_JOINT: 0.0}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sent_actions.append(dict(action))
        if self._on_step is not None and len(self.sent_actions) == self._on_step[0]:
            self._on_step[1]()
        return dict(action)

    def disconnect(self) -> None:
        return None


class _ChunkPolicy(Policy):
    """Emits a four-action chunk per query, so a rollout accrues steps fast."""

    @property
    def provider_name(self) -> str:
        return "test"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        return None

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{_JOINT: 0.4}] * 4


def _rollout_json(result: dict[str, Any]) -> dict[str, Any]:
    """Pull ``run_policy``'s structured report out of its tool-shaped result."""
    return next(block["json"] for block in result["content"] if "json" in block)


def _make_robot(arm: _FakeArm) -> HwRobot:
    """Build a Robot bypassing hardware init (the pattern used across tests/)."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "fake_arm"
    hw.action_horizon = 4
    hw.data_config = None
    hw.control_frequency = 500.0
    hw.action_sleep_time = 1.0 / 500.0
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = arm

    def publish(observation: dict[str, Any], skip_images: bool = False) -> None:
        return None

    hw._publish_ros_telemetry = publish  # type: ignore[method-assign]
    return hw


@pytest.fixture
def arm() -> _FakeArm:
    return _FakeArm()


@pytest.fixture
def hw(arm: _FakeArm) -> Any:
    robot = _make_robot(arm)
    yield robot
    robot._executor.shutdown(wait=False)


# Every public way to ask a Robot to drive a task, with the name each reports.
_ENTRY_POINTS = [
    ("run_policy", lambda hw: hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0, n_steps=4)),
    ("start_task", lambda hw: hw.start_task("pick the cube", policy_port=5555, duration=5.0)),
    ("execute_task", lambda hw: hw._execute_task_sync("pick the cube", policy_port=5555, duration=5.0)),
]


class TestAShutDownRobotRefusesTaskWork:
    """Every task entry point refuses a robot whose resources are already gone."""

    @pytest.mark.parametrize("method,call", _ENTRY_POINTS, ids=[name for name, _ in _ENTRY_POINTS])
    def test_a_task_started_after_cleanup_is_refused_and_commands_nothing(
        self, hw: HwRobot, arm: _FakeArm, method: str, call: Any
    ) -> None:
        """The refusal is tool-shaped, names the method and the cause, and moves no servo.

        ``start_task`` is the one that used to raise instead of returning: its
        submit reaches the dead executor. All three must report identically.
        """
        hw.cleanup()

        result = call(hw)

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert method in text
        assert "shut down" in text
        assert "cleanup()" in text
        assert arm.sent_actions == []

    def test_a_refused_call_does_not_take_the_motors_bus_claim(self, hw: HwRobot) -> None:
        """A refusal that claimed the bus would leave the robot refusing every later task.

        The claim is released in ``_drive_claimed_task``'s ``finally``, which a
        refused call never reaches, so the guard has to run before the claim.
        """
        hw.cleanup()

        assert hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0)["status"] == "error"

        assert hw._task_claimed is False

    def test_a_refused_call_leaves_the_previous_rollout_report_intact(self, hw: HwRobot, arm: _FakeArm) -> None:
        """A refusal reports on the call, not on the task state, so history survives it."""
        first = hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0, n_steps=8)
        assert first["status"] == "success"
        assert _rollout_json(first)["steps"] == 8

        hw.cleanup()
        hw.run_policy(_ChunkPolicy(), "pick it again", duration=5.0, n_steps=8)

        assert hw._task_state.step_count == 8
        assert hw._task_state.instruction == "pick the cube"
        assert len(arm.sent_actions) == 8


class TestAShutdownIsReportedAsAnInterruption:
    """The terminal block consults both latches its own loop condition honors."""

    def test_cleanup_latches_the_shutdown_without_recording_a_stop(self, hw: HwRobot) -> None:
        """This is why the truncation below cannot be detected from the stop latch.

        ``cleanup`` sets ``_shutdown_event`` first and only calls ``stop_task``
        for a rollout it still finds ``RUNNING``. A loop that has already exited
        on the latch has left ``RUNNING``, so nothing records the interruption
        anywhere except the latch itself.
        """
        hw._task_state.status = TaskStatus.COMPLETED

        hw.cleanup()

        assert hw._shutdown_event.is_set() is True
        assert hw._stop_requested.is_set() is False

    def test_a_rollout_truncated_by_a_shutdown_is_reported_stopped(self) -> None:
        """A rollout cut short at 8 of a 5-second budget is an interruption, not a success."""
        arm = _FakeArm()
        hw = _make_robot(arm)
        arm._on_step = (8, hw._shutdown_event.set)

        result = hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0)

        assert result["status"] == "error"
        assert _rollout_json(result)["status"] == TaskStatus.STOPPED.value
        assert hw._task_state.status is TaskStatus.STOPPED
        # The interruption was invisible to the stop latch - only the shutdown
        # latch recorded it, which is exactly what used to be ignored.
        assert hw._stop_requested.is_set() is False
        assert len(arm.sent_actions) < 100
        hw._executor.shutdown(wait=False)

    def test_an_explicit_stop_outranks_a_shutdown_in_the_report(self) -> None:
        """Both latches set means an operator pressed stop, which is the more specific cause."""
        arm = _FakeArm()
        hw = _make_robot(arm)

        def both() -> None:
            hw.stop_task()
            hw._shutdown_event.set()

        arm._on_step = (8, both)

        hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0)

        assert hw._task_state.status is TaskStatus.STOPPED
        hw._executor.shutdown(wait=False)

    def test_a_rollout_that_reaches_its_own_budget_still_completes(self, hw: HwRobot, arm: _FakeArm) -> None:
        """Negative control: the guard must not turn an ordinary success into an error."""
        result = hw.run_policy(_ChunkPolicy(), "pick the cube", duration=5.0, n_steps=8)

        assert result["status"] == "success"
        assert hw._task_state.status is TaskStatus.COMPLETED
        assert len(arm.sent_actions) == 8
