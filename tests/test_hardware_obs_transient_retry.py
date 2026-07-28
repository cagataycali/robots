# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""One dropped USB frame must not abort a whole rollout.

``observation = await asyncio.to_thread(self.robot.get_observation)`` sat bare
inside the loop's single top-level ``try``, so ANY read failure jumped to the
terminal ``except``, set ``ERROR`` and returned mid-manoeuvre with the arm still
torqued at its last commanded pose. Measured: a single ``TimeoutError`` on the 3rd
read ended a 40-step rollout at 8 steps.

Those exceptions are routine, not exceptional. The installed lerobot
``OpenCVCamera.read_latest`` raises:

* ``TimeoutError`` when the newest frame is older than ``max_age_ms`` (500 ms), and
* ``RuntimeError`` when its background read thread has died,

both expected under USB2 bandwidth contention - precisely what two MJPG streams on
one controller produce. The user's own live scripts hand-roll a 3x retry with
last-good-frame reuse and a consecutive-failure cap, which is the clearest evidence
the library should be doing it.

The read is now bounded-retry with last-good reuse, capped so the policy can never
be driven open-loop on a stale frame indefinitely.

No serial port is opened and no camera is touched.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class _FlakyRobot:
    """A driver whose get_observation fails on chosen call numbers."""

    def __init__(self, fail_on: set[int] | None = None, error: BaseException | None = None) -> None:
        self._fail_on = fail_on or set()
        self._error = error or TimeoutError("Cannot enable. Maybe the USB cable is bad?")
        self.calls = 0
        self.sent: list[dict[str, float]] = []
        self.disconnected = False

    def connect(self, calibrate: bool = False) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        self.calls += 1
        if self.calls in self._fail_on:
            raise self._error
        return dict.fromkeys(_KEYS, float(self.calls))

    def send_action(self, action: dict[str, float]) -> None:
        self.sent.append(action)

    def disconnect(self) -> None:
        self.disconnected = True


class _ChunkPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        self.seen.append(dict(observation))
        return [dict.fromkeys(_KEYS, 0.0) for _ in range(8)]


def _robot(driver: _FlakyRobot, **overrides) -> Robot:
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
    hw._obs_retries = overrides.get("obs_retries", 3)
    hw._obs_retry_backoff_s = overrides.get("backoff", 0.0)
    hw._max_consecutive_obs_failures = overrides.get("max_failures", 20)
    return hw


def _run(hw: Robot, policy: Policy, n_steps: int = 40) -> None:
    asyncio.run(hw._execute_task_async("t", policy_object=policy, n_steps=n_steps, duration=30.0))


class TestTransientFailuresAreSurvived:
    def test_a_single_dropped_frame_does_not_abort(self):
        """The regression: this ended a 40-step rollout at 8 steps."""
        driver = _FlakyRobot(fail_on={3})
        hw = _robot(driver)

        _run(hw, _ChunkPolicy())

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._task_state.step_count == 40

    def test_a_burst_of_failures_is_survived(self):
        driver = _FlakyRobot(fail_on={3, 4, 5})
        hw = _robot(driver)

        _run(hw, _ChunkPolicy())

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._task_state.step_count == 40

    @pytest.mark.parametrize(
        "error",
        [
            TimeoutError("latest frame is too old: 812.4 ms (max allowed: 500 ms)."),
            RuntimeError("read thread is not running."),
            OSError("device re-enumerated"),
        ],
    )
    def test_each_documented_transient_type_is_tolerated(self, error):
        """lerobot raises TimeoutError/RuntimeError; USB re-enumeration gives OSError."""
        driver = _FlakyRobot(fail_on={2}, error=error)
        hw = _robot(driver)

        _run(hw, _ChunkPolicy(), n_steps=16)

        assert hw._task_state.status is TaskStatus.COMPLETED

    def test_retries_happen_within_the_step_before_reusing(self):
        """The retry budget is spent first; a single failure needs no stale frame."""
        driver = _FlakyRobot(fail_on={3})
        hw = _robot(driver)

        _run(hw, _ChunkPolicy(), n_steps=16)

        # Call 3 failed once, and the retry on the same step succeeded.
        assert hw._consecutive_obs_failures == 0


class TestPermanentFailureStopsCleanly:
    def test_exhausting_the_budget_ends_the_rollout_with_an_error(self):
        driver = _FlakyRobot(fail_on=set(range(2, 500)))
        hw = _robot(driver, max_failures=5)

        _run(hw, _ChunkPolicy())

        assert hw._task_state.status is TaskStatus.ERROR
        message = hw._task_state.error_message
        assert "consecutive" in message
        assert "stale frame" in message

    def test_the_cap_is_honoured(self):
        """A stale frame must not be replayed forever - that is open-loop motion."""
        driver = _FlakyRobot(fail_on=set(range(2, 500)))
        hw = _robot(driver, max_failures=4)
        policy = _ChunkPolicy()

        _run(hw, policy)

        # 1 good observation + at most cap reused ones reach the policy.
        assert len(policy.seen) <= 1 + 4

    def test_a_failure_on_the_very_first_read_stops_immediately(self):
        """With no previous frame there is nothing to reuse."""
        driver = _FlakyRobot(fail_on=set(range(1, 500)))
        hw = _robot(driver)

        _run(hw, _ChunkPolicy())

        assert hw._task_state.status is TaskStatus.ERROR
        assert driver.sent == [], "the arm moved without ever having a valid observation"

    def test_error_message_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        driver = _FlakyRobot(fail_on=set(range(2, 500)))
        hw = _robot(driver, max_failures=3)

        _run(hw, _ChunkPolicy())

        assert hw._task_state.error_message.isascii()


class TestBudgetAccounting:
    def test_a_good_read_resets_the_consecutive_counter(self):
        """Intermittent drops must not accumulate toward the cap across a rollout."""
        # Fail one read every few steps, more times than the cap, but never
        # consecutively enough to exhaust it.
        driver = _FlakyRobot(fail_on={2, 4, 6, 8, 10, 12})
        hw = _robot(driver, max_failures=2)

        _run(hw, _ChunkPolicy(), n_steps=40)

        assert hw._task_state.status is TaskStatus.COMPLETED

    def test_state_is_reset_between_rollouts(self):
        """A previous rollout's failure count must not shorten the next one."""
        driver = _FlakyRobot(fail_on=set(range(2, 500)))
        hw = _robot(driver, max_failures=3)
        _run(hw, _ChunkPolicy())
        assert hw._task_state.status is TaskStatus.ERROR

        hw.robot = _FlakyRobot()  # a healthy driver for the second run
        _run(hw, _ChunkPolicy(), n_steps=8)

        assert hw._task_state.status is TaskStatus.COMPLETED
