# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A shut-down Robot must refuse work, not report a no-op as success.

``cleanup()`` / ``stop()`` set ``_shutdown_event`` and never clear it, and the
control loop's condition includes ``not _shutdown_event.is_set()``. So a rollout
started afterwards exited before its FIRST iteration while the terminal block still
performed the ``RUNNING -> COMPLETED`` transition, and ``run_policy`` maps
``COMPLETED`` to ``status="success"``.

Measured::

    run 1 status: success | sent: 8
    run 2 status: success
    run 2 text  : Policy rollout completed: 0 steps in 0.0s
    run 2 sent  : 0

The agent was told the rollout completed while the arm never moved. ``start_task``
was worse: a bare ``RuntimeError: cannot schedule new futures after shutdown`` from
the dead executor.

Two independent guards now cover it: the entry points refuse a shut-down Robot
explicitly, and a zero-step rollout is never reported as COMPLETED regardless of
why the loop exited.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class _Driver:
    def __init__(self) -> None:
        self.sent: list[dict[str, float]] = []
        self.disconnected = False
        self.action_features = dict.fromkeys(_KEYS, float)

    def connect(self, calibrate: bool = False) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        return dict.fromkeys(_KEYS, 0.0)

    def send_action(self, action: dict[str, float]):
        self.sent.append(action)
        return dict(action)

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


def _robot(driver: _Driver) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = 50.0
    hw.action_sleep_time = 0.001
    hw.action_horizon = 8
    hw.robot = driver
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_state = hardware_robot.RobotTaskState()
    hw.mesh = None
    hw.peer_id = None
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm")
    hw._publish_ros_telemetry = lambda observation: None  # type: ignore[assignment,method-assign,misc]
    hw._obs_retries = 3
    hw._obs_retry_backoff_s = 0.0
    hw._max_consecutive_obs_failures = 20
    hw._dropped_action_steps = 0
    hw._dropped_action_keys = []
    return hw


class TestEntryPointsRefuseAfterShutdown:
    def test_run_policy_reports_an_error_naming_the_shutdown(self):
        """The regression: this returned success with 0 steps."""
        driver = _Driver()
        hw = _robot(driver)
        hw._shutdown_event.set()  # what cleanup()/stop() does

        result = hw.run_policy(policy_object=_ChunkPolicy(), instruction="t", n_steps=8)

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "shut down" in text
        assert "construct a new Robot" in text
        assert driver.sent == [], "the arm was commanded after shutdown"

    def test_start_task_returns_a_structured_error_not_a_raise(self):
        """A dead executor raised RuntimeError straight out of the tool call."""
        hw = _robot(_Driver())
        hw._executor.shutdown(wait=True)
        hw._shutdown_event.set()

        result = hw.start_task("t", policy_port=9000)

        assert result["status"] == "error"
        assert "shut down" in result["content"][0]["text"]

    def test_a_healthy_robot_is_not_gated(self):
        driver = _Driver()
        hw = _robot(driver)

        result = hw.run_policy(policy_object=_ChunkPolicy(), instruction="t", n_steps=8)

        assert result["status"] == "success", result
        assert len(driver.sent) == 8

    def test_error_text_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        hw = _robot(_Driver())
        hw._shutdown_event.set()

        result = hw.run_policy(policy_object=_ChunkPolicy(), instruction="t", n_steps=8)

        assert result["content"][0]["text"].isascii()

    def test_guard_tolerates_a_half_built_robot(self):
        """A Robot whose __init__ did not complete has no latch to read."""
        hw = Robot.__new__(Robot)
        hw.tool_name_str = "half_built"

        assert hw._shutdown_guard("run_policy") is None


class TestZeroStepRolloutIsNotCompleted:
    def test_a_rollout_that_applies_no_action_reports_error(self):
        """Independent of WHY the loop exited: no motion is not a completed task."""
        driver = _Driver()
        hw = _robot(driver)
        # n_steps=0 makes the inner loop break before the first send_action.
        hw._execute_task_sync("t", policy_object=_ChunkPolicy(), n_steps=0, duration=5.0)

        assert hw._task_state.status is TaskStatus.ERROR
        assert "0 actions" in hw._task_state.error_message
        assert driver.sent == []

    def test_the_zero_step_message_is_actionable(self):
        hw = _robot(_Driver())
        hw._execute_task_sync("t", policy_object=_ChunkPolicy(), n_steps=0, duration=5.0)

        message = hw._task_state.error_message
        assert "duration/n_steps" in message
        assert "shut down" in message
        assert message.isascii()

    def test_a_latched_shutdown_mid_object_still_reports_error_not_completed(self):
        """Belt and braces: even bypassing the entry guard, 0 steps is not success."""
        driver = _Driver()
        hw = _robot(driver)
        hw._shutdown_event.set()

        # Call the loop directly, past the run_policy guard.
        hw._execute_task_sync("t", policy_object=_ChunkPolicy(), n_steps=8, duration=5.0)

        assert hw._task_state.status is TaskStatus.ERROR
        assert driver.sent == []

    def test_a_normal_rollout_still_completes(self):
        driver = _Driver()
        hw = _robot(driver)

        hw._execute_task_sync("t", policy_object=_ChunkPolicy(), n_steps=4, duration=5.0)

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._task_state.step_count == 4


class TestScopeCorrections:
    def test_gc_of_a_separate_robot_does_not_latch_this_one(self):
        """Per the ledger's verifier: __del__ runs per-object, not globally.

        The original finding claimed the latch could fire from garbage collection
        of any transient reference. It cannot - only an explicit cleanup()/stop()
        on THIS object latches it.
        """
        import gc

        keeper = _robot(_Driver())
        transient = _robot(_Driver())
        transient._shutdown_event.set()
        del transient
        gc.collect()

        assert not keeper._shutdown_event.is_set()
        assert keeper._shutdown_guard("run_policy") is None
