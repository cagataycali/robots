"""The hardware control loop must actually run at its configured rate.

``_execute_task_async`` previously slept a fixed ``1/control_frequency`` after
every ``send_action``, so the observation read and the policy inference were
ADDITIVE to every control period. A loop configured for 50Hz with a 33ms
two-camera observation read and 120ms of inference really ran at ~23Hz - less
than half the declared rate.

That is not merely slow. The loop tells the policy ``control_frequency`` before
the rollout (RTC-capable policies convert their inference latency into a step
count against it and blend chunk seams accordingly), so a rate the loop never
achieves silently corrupts every chunk-seam blend.

These tests drive the REAL loop with a fake robot/policy whose only behaviour is
to consume a known amount of time, then assert on the achieved rate. No serial
port is opened and no arm is commanded.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from strands_robots.hardware_robot import Robot, RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_KEYS = [f"{m}.pos" for m in _MOTORS]


class _FakeRobot:
    """A driver whose reads and writes cost a fixed, known amount of time."""

    def __init__(self, observation_s: float = 0.0, send_s: float = 0.0) -> None:
        self._observation_s = observation_s
        self._send_s = send_s
        self.sent: list[float] = []

    def get_observation(self) -> dict[str, float]:
        time.sleep(self._observation_s)
        return dict.fromkeys(_KEYS, 0.0)

    def send_action(self, action: dict[str, float]) -> None:
        time.sleep(self._send_s)
        self.sent.append(time.perf_counter())

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = False) -> None:
        pass


class _FakePolicy(Policy):
    """A chunk-emitting policy whose inference costs a fixed amount of time."""

    def __init__(self, inference_s: float = 0.0, chunk: int = 8) -> None:
        super().__init__()
        self._inference_s = inference_s
        self._chunk = chunk
        self.told_frequency: float | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_robot_state_keys(self, keys: list[str]) -> None:
        pass

    def set_control_frequency(self, hz: float) -> None:
        self.told_frequency = hz
        super().set_control_frequency(hz)

    async def get_actions(self, observation, instruction, **kwargs):
        await asyncio.sleep(self._inference_s)
        return [dict.fromkeys(_KEYS, 0.0) for _ in range(self._chunk)]

    @property
    def execution_horizon(self) -> int:
        return self._chunk


def _make_robot(fake_robot: _FakeRobot, control_frequency: float) -> Robot:
    """Build a Robot bound to a fake driver, skipping lerobot construction."""
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = control_frequency
    hw.action_sleep_time = 1.0 / control_frequency
    hw.action_horizon = 8
    hw.robot = fake_robot
    hw._shutdown_event = threading.Event()
    hw._task_state = RobotTaskState()
    hw.mesh = None
    hw.peer_id = None
    # A real executor so Robot.cleanup() (reached via __del__) has something to
    # shut down; a None here makes teardown log a spurious cleanup error.
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm_executor")
    hw._publish_ros_telemetry = lambda observation: None  # type: ignore[assignment,method-assign,misc]

    async def _connect():
        return True, "ok"

    hw._connect_robot = _connect  # type: ignore[method-assign]

    async def _init_policy(policy):
        policy.set_control_frequency(hw.control_frequency)
        return True

    hw._initialize_policy = _init_policy  # type: ignore[method-assign]
    return hw


def _run(hw: Robot, policy: _FakePolicy, duration: float) -> float:
    """Run one rollout and return the wall-clock elapsed time."""
    started = time.perf_counter()
    asyncio.run(hw._execute_task_async("t", policy_object=policy, duration=duration))
    return time.perf_counter() - started


class TestAchievesConfiguredRate:
    def test_per_observation_cost_is_absorbed_not_added(self):
        """Regression: observation + inference must not stretch the control period.

        Pre-fix this configuration achieved ~23Hz of a configured 50Hz. The
        deadline schedule absorbs the per-chunk cost into the periods it overlaps,
        so the loop tracks its configured rate.
        """
        fake = _FakeRobot(observation_s=0.033, send_s=0.004)
        hw = _make_robot(fake, control_frequency=50.0)
        policy = _FakePolicy(inference_s=0.120, chunk=8)

        elapsed = _run(hw, policy, duration=3.0)

        achieved = hw._task_state.step_count / elapsed
        # The policy must have been told the configured rate ...
        assert policy.told_frequency == 50.0
        # ... and the loop must actually deliver close to it. Pre-fix: ~23Hz.
        assert achieved >= 0.8 * 50.0, f"achieved {achieved:.1f}Hz of 50Hz"
        # The loop must never OVERSHOOT its rate either: missed deadlines are
        # absorbed, not repaid as a burst of unsleeping servo writes.
        assert achieved <= 1.1 * 50.0, f"achieved {achieved:.1f}Hz exceeds 50Hz"

    def test_zero_latency_loop_tracks_its_period(self):
        """With no driver/inference cost the rate is bounded by the period alone."""
        fake = _FakeRobot()
        hw = _make_robot(fake, control_frequency=25.0)
        policy = _FakePolicy(chunk=4)

        elapsed = _run(hw, policy, duration=2.0)

        achieved = hw._task_state.step_count / elapsed
        assert 0.8 * 25.0 <= achieved <= 1.1 * 25.0, f"achieved {achieved:.1f}Hz of 25Hz"

    def test_achieved_rate_is_recorded_and_reported(self):
        """The real rate must be observable, not left implicit."""
        fake = _FakeRobot(observation_s=0.01)
        hw = _make_robot(fake, control_frequency=50.0)

        _run(hw, _FakePolicy(chunk=8), duration=1.0)

        assert hw._task_state.status == TaskStatus.COMPLETED
        assert hw._task_state.achieved_hz > 0
        text = hw.get_task_status()["content"][0]["text"]
        assert "Control Rate:" in text
        assert "achieved" in text and "configured" in text


class TestUnachievableRateIsReported:
    def test_warns_when_the_loop_cannot_keep_up(self, caplog):
        """A rate the hardware cannot sustain must be named, not silently missed.

        Inference alone (200ms) exceeds a whole 1-action chunk at 50Hz (20ms), so
        no scheduling can reach the configured rate. The operator has to be told,
        because the policy was handed the declared rate for RTC blending.
        """
        fake = _FakeRobot()
        hw = _make_robot(fake, control_frequency=50.0)
        policy = _FakePolicy(inference_s=0.200, chunk=1)

        with caplog.at_level(logging.WARNING):
            _run(hw, policy, duration=2.0)

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("is configured for 50.0Hz" in m for m in msgs), msgs
        assert any("never achieved" in m for m in msgs), msgs

    def test_no_warning_when_the_rate_is_met(self, caplog):
        fake = _FakeRobot()
        hw = _make_robot(fake, control_frequency=20.0)

        with caplog.at_level(logging.WARNING):
            _run(hw, _FakePolicy(chunk=8), duration=1.5)

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("configured for" in m for m in msgs), msgs

    @pytest.mark.parametrize("steps", [0, 1])
    def test_degenerate_rollouts_do_not_warn(self, steps):
        """A rollout with no measurable rate must not emit a bogus shortfall."""
        fake = _FakeRobot()
        hw = _make_robot(fake, control_frequency=50.0)
        with pytest.MonkeyPatch.context() as mp:
            recorded: list[str] = []
            mp.setattr(
                "strands_robots.hardware_robot.logger.warning",
                lambda *a, **k: recorded.append(str(a[0])),
            )
            hw._warn_on_rate_shortfall(elapsed=1.0, steps=steps)
        assert recorded == []
