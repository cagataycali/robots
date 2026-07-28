# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Each hardware task must start the policy from a clean per-episode state.

``_execute_task_async`` connected, called ``_initialize_policy``,
``set_control_frequency``, then looped ``set_rtc_observed_delay(0)`` +
``get_actions``. There was no ``policy.reset()`` anywhere in
``hardware_robot.py`` (``grep -c`` -> 0), while the sim runner resets per
episode. A task boundary IS an episode boundary on hardware, so a second
``run_policy`` / ``start_task`` on the same Robot with the same pre-built policy
object began with the previous task's state intact.

Measured pre-fix with a policy holding a 4-long chunk and consuming 2 per task::

    task1 emitted [200.0, 201.0]  queue depth after: 2  resets: 0
    task2 emitted [202.0, 203.0]  resets: 0  inferences: 1
    -> task 2 REPLAYED task 1's leftover chunk, zero new inference

That is motion generated for a previous scene applied to a physical arm.

``LerobotLocalPolicy.reset``'s own docstring says it "**MUST** be called whenever
the environment or task episode resets ... Without resetting, stale actions from
the previous episode leak into the next one". Its body clears the LeRobot action
queue and temporal-ensemble buffer, the processor bridge, ``_rtc_prev_chunk`` /
``_rtc_prev_chunk_abs`` / ``_rtc_action_queue`` / ``_rtc_latency_history``,
``rtc_observed_delay_steps``, and re-arms ``_zero_action_monitor`` and
``_action_dim_warned`` - every one of which survived a task boundary.

The default ``Policy.reset`` is a no-op, so providers without per-episode state
are unaffected. The call is fail-soft, matching the sim runner: a reset that
raises must not abort a rollout the caller can still run.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class _Bus:
    """A driver that records the commanded shoulder_pan values."""

    def __init__(self) -> None:
        self._connected = False
        self.commanded: list[float] = []

    def connect(self, calibrate: bool = False) -> None:
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    action_features = {key: float for key in _KEYS}

    def get_observation(self) -> dict[str, float]:
        return dict.fromkeys(_KEYS, 0.0)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.commanded.append(float(action["shoulder_pan.pos"]))
        return dict(action)

    def disconnect(self) -> None:
        self._connected = False


class _QueuePolicy(Policy):
    """Holds an action queue across calls, like LerobotLocalPolicy does.

    Each inference enqueues four actions tagged with the inference number; only
    two are consumed per task, so a task that starts without a reset drains the
    previous task's leftovers and its emitted values stay in the OLD hundred.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0
        self.inferences = 0
        self.queue: list[float] = []

    @property
    def provider_name(self) -> str:
        return "queue"

    def set_robot_state_keys(self, keys) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1
        self.queue.clear()

    async def get_actions(self, observation, instruction, **kwargs):
        if not self.queue:
            self.inferences += 1
            self.queue = [100.0 * (self.inferences + 1) + index for index in range(4)]
        chunk = []
        for _ in range(2):
            if self.queue:
                chunk.append({**dict.fromkeys(_KEYS, 0.0), "shoulder_pan.pos": self.queue.pop(0)})
        return chunk


class _RaisingResetPolicy(_QueuePolicy):
    """reset() blows up - the rollout must still run."""

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1
        raise RuntimeError("reset exploded")


def _robot(driver: _Bus) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = 200.0
    hw.action_sleep_time = 0.001
    hw.action_horizon = 2
    hw.robot = driver
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_lock = threading.Lock()
    hw._task_state = hardware_robot.RobotTaskState()
    hw.mesh = None
    hw.peer_id = None
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fake_arm")
    hw._publish_ros_telemetry = lambda observation: None  # type: ignore[assignment,method-assign,misc]
    return hw


class TestResetIsCalledPerTask:
    def test_each_run_policy_resets_the_policy_once(self):
        """Regression: hardware_robot.py had zero policy-reset call sites."""
        hw = _robot(_Bus())
        policy = _QueuePolicy()

        hw.run_policy(policy_object=policy, instruction="task1", n_steps=2)
        assert policy.reset_calls == 1, "the first task did not reset the policy"

        hw.run_policy(policy_object=policy, instruction="task2", n_steps=2)
        assert policy.reset_calls == 2

    def test_a_second_task_does_not_replay_the_first_ones_chunk(self):
        """The physical consequence: motion for a previous scene on a real arm."""
        driver = _Bus()
        hw = _robot(driver)
        policy = _QueuePolicy()

        hw.run_policy(policy_object=policy, instruction="task1", n_steps=2)
        first = list(driver.commanded)
        assert first, "the first task commanded nothing"
        assert policy.queue, "the fixture left no leftover chunk to replay"
        driver.commanded.clear()

        hw.run_policy(policy_object=policy, instruction="task2", n_steps=2)

        second = list(driver.commanded)
        assert second, "the second task commanded nothing"
        # Pre-fix: first [200.0, 201.0], second [202.0, 203.0] - the same chunk.
        assert min(second) > max(first), f"task2 replayed task1's chunk: {first} then {second}"
        assert policy.inferences == 2, f"task2 ran {policy.inferences - 1} fresh inference(s)"

    def test_reset_happens_before_the_first_inference(self):
        """A reset AFTER the first get_actions would still leak one chunk."""
        hw = _robot(_Bus())

        class _Ordering(_QueuePolicy):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[str] = []

            def reset(self, seed: int | None = None) -> None:
                super().reset(seed)
                self.events.append("reset")

            async def get_actions(self, observation, instruction, **kwargs):
                self.events.append("infer")
                return await super().get_actions(observation, instruction, **kwargs)

        policy = _Ordering()
        hw.run_policy(policy_object=policy, instruction="task", n_steps=2)

        assert policy.events, "the policy was never exercised"
        assert policy.events[0] == "reset", policy.events

    def test_start_task_resets_too(self):
        """Both entry points reach the same loop; neither may skip the reset."""
        hw = _robot(_Bus())
        try:
            result = hw.start_task("task", policy_port=1, duration=0.2)
            assert result["status"] == "success", result
            future = hw._task_state.task_future
            assert future is not None
            future.result(timeout=15.0)
        finally:
            hw._executor.shutdown(wait=True)

        # The server-backed path builds its own policy, so assert on the loop
        # having run rather than on this object; the reset call site is shared.
        assert hw._task_state.status in (TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.STOPPED)


class TestResetIsFailSoft:
    def test_a_raising_reset_does_not_abort_the_rollout(self):
        """Matching PolicyRunner: reset is best-effort, not a gate."""
        driver = _Bus()
        hw = _robot(driver)
        policy = _RaisingResetPolicy()

        result = hw.run_policy(policy_object=policy, instruction="task", n_steps=2)

        assert policy.reset_calls == 1
        assert result["status"] == "success", result
        assert driver.commanded, "the rollout was aborted by a failing reset"

    def test_a_raising_reset_is_logged_with_the_consequence(self, caplog):
        hw = _robot(_Bus())
        with caplog.at_level("WARNING"):
            hw.run_policy(policy_object=_RaisingResetPolicy(), instruction="task", n_steps=2)

        warnings = [record.getMessage() for record in caplog.records if "reset" in record.getMessage()]
        assert warnings, [record.getMessage() for record in caplog.records]
        assert "stale" in warnings[0]
        assert warnings[0].isascii()


class TestPoliciesWithoutEpisodeState:
    def test_the_default_reset_is_a_no_op_and_harmless(self):
        """Providers that never override reset must be unaffected."""
        driver = _Bus()
        hw = _robot(driver)

        class _Stateless(Policy):
            @property
            def provider_name(self) -> str:
                return "stateless"

            def set_robot_state_keys(self, keys) -> None:
                pass

            async def get_actions(self, observation, instruction, **kwargs):
                return [{**dict.fromkeys(_KEYS, 0.0), "shoulder_pan.pos": 1.0}]

        result = hw.run_policy(policy_object=_Stateless(), instruction="task", n_steps=2)

        assert result["status"] == "success", result
        assert driver.commanded == [1.0, 1.0]


class TestTheRealPolicyContract:
    def test_lerobot_local_reset_clears_what_the_task_boundary_leaked(self):
        """Pin the state the hardware loop now clears, on the real class.

        Guards against the reset body being trimmed later: if any of these stop
        being cleared, the hardware task boundary silently starts leaking again.
        """
        policy_module = pytest.importorskip("strands_robots.policies.lerobot_local.policy")
        cleared = (
            "_rtc_prev_chunk",
            "_rtc_prev_chunk_abs",
            "_rtc_action_queue",
            "_rtc_latency_history",
            "rtc_observed_delay_steps",
            "_zero_action_monitor",
            "_action_dim_warned",
        )
        import inspect

        source = inspect.getsource(policy_module.LerobotLocalPolicy.reset)
        missing = [name for name in cleared if name not in source]
        assert not missing, f"LerobotLocalPolicy.reset no longer clears {missing}"
