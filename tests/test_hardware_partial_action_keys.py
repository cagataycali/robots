# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A driver that refuses some action keys must not be reported as success.

The loop did ``await asyncio.to_thread(self.robot.send_action, action_dict)`` and
DISCARDED the return value, then incremented ``step_count`` unconditionally. But
lerobot drivers return the action they actually sent - ``SOFollower.send_action``'s
docstring says "this function always returns the action actually sent" and its body
filters ``if key.endswith(".pos")``.

So an action naming some motors correctly and some incorrectly reported
``status="success"`` with a full step count while only the matching subset moved:
half the arm frozen, silently, for the whole rollout.

Scope. A TOTAL mismatch is already loud (``MotorsBus.sync_write({})`` raises), and
``_derive_robot_state_keys`` now binds from ``action_features``, so the residual
case is the PARTIAL one - which can only be detected from the driver's own answer,
which is precisely what the discarded return value was.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_GOOD_KEYS = [f"{m}.pos" for m in _MOTORS]
_WRONG_KEYS = ["joint_3", "joint_4", "joint_5"]


class _FilteringDriver:
    """Echoes only the keys it recognises, like every real lerobot driver."""

    def __init__(self, accepts: list[str] | None = None, echo: bool = True) -> None:
        self.action_features = dict.fromkeys(accepts if accepts is not None else _GOOD_KEYS, float)
        self._echo = echo
        self.writes: list[dict[str, float]] = []

    def connect(self, calibrate: bool = False) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        return dict.fromkeys(self.action_features, 0.0)

    def send_action(self, action: dict[str, float]):
        sent = {k: v for k, v in action.items() if k in self.action_features}
        self.writes.append(sent)
        return sent if self._echo else None


class _FixedKeysPolicy(Policy):
    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = keys

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(self._keys, 0.0) for _ in range(4)]


def _robot(driver: _FilteringDriver) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = 50.0
    hw.action_sleep_time = 0.001
    hw.action_horizon = 4
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


def _run(hw: Robot, policy: Policy, n_steps: int = 8) -> None:
    asyncio.run(hw._execute_task_async("t", policy_object=policy, n_steps=n_steps, duration=10.0))


class TestPartialMismatchIsNotSuccess:
    def test_every_step_dropping_keys_reports_error(self):
        """The regression: status='success' with half the arm frozen."""
        driver = _FilteringDriver()
        hw = _robot(driver)

        _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS))

        assert hw._task_state.status is TaskStatus.ERROR
        assert "refused to command" in hw._task_state.error_message
        # The arm really did only get half its joints.
        assert len(driver.writes[0]) == 3

    def test_the_warning_names_both_key_sets(self):
        driver = _FilteringDriver()
        hw = _robot(driver)

        with pytest.MonkeyPatch.context():
            records: list[str] = []
            handler = logging.Handler()
            handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
            logger = logging.getLogger("strands_robots.hardware_robot")
            logger.addHandler(handler)
            try:
                _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS))
            finally:
                logger.removeHandler(handler)

        joined = " ".join(records)
        assert "joint_3" in joined, joined
        assert "shoulder_pan.pos" in joined, joined
        assert "NOT moving" in joined, joined

    def test_the_warning_fires_only_once(self):
        """A per-step warning would drown the log over a long rollout."""
        driver = _FilteringDriver()
        hw = _robot(driver)

        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
        logger = logging.getLogger("strands_robots.hardware_robot")
        logger.addHandler(handler)
        try:
            _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS), n_steps=16)
        finally:
            logger.removeHandler(handler)

        assert sum("did not command" in m for m in records) == 1

    def test_status_text_surfaces_the_drop(self):
        driver = _FilteringDriver()
        hw = _robot(driver)
        _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS))

        text = hw.get_task_status()["content"][0]["text"]

        assert "Dropped Actions:" in text

    def test_error_message_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        hw = _robot(_FilteringDriver())
        _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS))

        assert hw._task_state.error_message.isascii()


class TestFullyMatchingActionsAreUnaffected:
    def test_all_keys_accepted_completes_cleanly(self):
        driver = _FilteringDriver()
        hw = _robot(driver)

        _run(hw, _FixedKeysPolicy(_GOOD_KEYS))

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._dropped_action_steps == 0
        assert "Dropped Actions" not in hw.get_task_status()["content"][0]["text"]

    def test_a_driver_returning_none_is_not_treated_as_a_drop(self):
        """Absence of evidence is not evidence of a drop; older drivers return None."""
        driver = _FilteringDriver(echo=False)
        hw = _robot(driver)

        _run(hw, _FixedKeysPolicy(_GOOD_KEYS))

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._dropped_action_steps == 0


class TestIntermittentDropsAreReportedButNotFatal:
    def test_some_steps_dropping_stays_completed_but_reports(self):
        """Only an every-step mismatch is a name disagreement worth failing on."""
        driver = _FilteringDriver()
        hw = _robot(driver)
        # Simulate an intermittent drop by pre-loading the counter below the
        # step count: the rollout itself sends fully-matching actions.
        _run(hw, _FixedKeysPolicy(_GOOD_KEYS))
        hw._dropped_action_steps = 1
        hw._dropped_action_keys = ["joint_9"]

        assert hw._task_state.status is TaskStatus.COMPLETED
        assert "joint_9" in hw._dropped_action_summary()

    def test_state_is_reset_between_rollouts(self):
        driver = _FilteringDriver()
        hw = _robot(driver)
        _run(hw, _FixedKeysPolicy(_GOOD_KEYS[:3] + _WRONG_KEYS))
        assert hw._dropped_action_steps > 0

        _run(hw, _FixedKeysPolicy(_GOOD_KEYS))

        assert hw._dropped_action_steps == 0
        assert hw._task_state.status is TaskStatus.COMPLETED
