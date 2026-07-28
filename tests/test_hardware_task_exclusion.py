# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Only one control loop may ever drive the motors bus.

``run_policy`` and ``start_task`` both guarded with
``if self._task_state.status == TaskStatus.RUNNING``. But ``_execute_task_async``
sets ``CONNECTING`` first and only reaches ``RUNNING`` after ``_connect_robot()`` -
a FeetechMotorsBus handshake plus per-camera warmup, 2-3 s on a real SO-101. A
second call inside that window passed the guard, and both loops then interleaved
``sync_read``/``sync_write`` on one ``FeetechMotorsBus``.

Measured against the real loop with a 1.5 s ``connect()``, two policies tagging
their own actions::

    status during 2nd call: connecting
    2nd run_policy accepted: success
    policies that reached the bus: ['B']      <- policy A's rollout was hijacked
    shared step_count: 8                      <- one _task_state, two writers

``FeetechMotorsBus`` is not thread-safe: ``sync_read``/``sync_write`` drive one
``port_handler`` whose ``is_using`` flag is the only interlock. Two interleaved
transactions on one half-duplex RS-485 bus give framing errors, or a
``Goal_Position`` write from policy A landing between policy B's read and write -
two different intents applied to one physical arm. The class already models
exclusivity (single-slot ``_task_state``, ``ThreadPoolExecutor(max_workers=1)``);
the guard checked the wrong predicate.

The fix is a real non-blocking ``threading.Lock`` held across the whole rollout in
``_execute_task_sync`` - the single funnel every entry point goes through - plus
``_task_busy_error`` at each entry point so the rejection is a clean tool error
rather than a race. ``start_task``'s outstanding future counts as busy too, since
it returns before its executor job has taken the lock.

``mesh/core.py`` routes ``{"action": "execute"|"start"}`` into the same entry
points, so two mesh peers could trigger this as well.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot, TaskStatus
from strands_robots.policies.base import Policy

_KEYS = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]

#: gripper.pos value -> owning policy. Policies are attributed by the VALUE they
#: command on a declared joint, so the fixture never sends an undeclared key.
_TAGS: dict[float, str] = {}


class _Bus:
    """A driver that records WHICH policy's actions reached the wire."""

    def __init__(self, connect_seconds: float = 1.0) -> None:
        self._connect_seconds = connect_seconds
        self._connected = False
        self.writers: list[str] = []
        #: Peak number of send_action calls in flight at once. >1 means two
        #: loops were mid-transaction on one half-duplex port.
        self.max_concurrent = 0
        self._in_flight = 0
        self._counter_lock = threading.Lock()

    def connect(self, calibrate: bool = False) -> None:
        time.sleep(self._connect_seconds)
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
        with self._counter_lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            # A real sync_write holds the port for a serial round trip; the
            # sleep is what makes an interleave observable.
            time.sleep(0.005)
            # Attribution rides in a DECLARED joint value, not an extra key: an
            # undeclared key would (correctly) trip the dropped-action guard.
            self.writers.append(_TAGS.get(action["gripper.pos"], "?"))
        finally:
            with self._counter_lock:
                self._in_flight -= 1
        return dict(action)

    def disconnect(self) -> None:
        self._connected = False


class _TaggingPolicy(Policy):
    """Tags every action with its own identity so the bus can attribute writes."""

    def __init__(self, who: str, chunk: int = 4) -> None:
        super().__init__()
        self.who = who
        self.chunk = chunk
        self.inferences = 0
        # A distinct, in-range gripper command per policy is the tag.
        self.tag = round(0.01 * (len(_TAGS) + 1), 4)
        _TAGS[self.tag] = who

    @property
    def provider_name(self) -> str:
        return self.who

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        self.inferences += 1
        return [{**dict.fromkeys(_KEYS, 0.0), "gripper.pos": self.tag} for _ in range(self.chunk)]


def _robot(driver: _Bus) -> Robot:
    hw = Robot.__new__(Robot)
    hw.tool_name_str = "fake_arm"
    hw.control_frequency = 50.0
    hw.action_sleep_time = 0.002
    hw.action_horizon = 4
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


def _text(result: dict) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []) if "text" in block)


class TestRunPolicyDuringConnect:
    def test_a_second_run_policy_is_rejected_while_connecting(self):
        """The regression: the second call was accepted mid-CONNECTING."""
        driver = _Bus(connect_seconds=1.0)
        hw = _robot(driver)
        first = _TaggingPolicy("A")
        second = _TaggingPolicy("B")
        outcome: dict = {}

        thread = threading.Thread(
            target=lambda: outcome.update(first_result=hw.run_policy(policy_object=first, instruction="a", n_steps=8)),
            daemon=True,
        )
        thread.start()
        try:
            time.sleep(0.3)  # firmly inside the CONNECTING window
            assert hw._task_state.status is TaskStatus.CONNECTING, "the fixture never reached CONNECTING"

            rejected = hw.run_policy(policy_object=second, instruction="b", n_steps=8)

            assert rejected["status"] == "error", rejected
            assert "Task already running" in _text(rejected)
            # The rejected policy must never have been asked for an action.
            assert second.inferences == 0
        finally:
            thread.join(timeout=30.0)
            assert not thread.is_alive(), "the first rollout never finished"

        # Only the first policy's intent reached the wire.
        assert set(driver.writers) == {"A"}, driver.writers
        assert driver.max_concurrent == 1, f"{driver.max_concurrent} concurrent bus transactions"

    def test_the_rejection_does_not_report_the_other_rollouts_steps(self):
        """A rejected caller must not be handed the running loop's step count."""
        driver = _Bus(connect_seconds=1.0)
        hw = _robot(driver)
        thread = threading.Thread(
            target=lambda: hw.run_policy(policy_object=_TaggingPolicy("A"), instruction="a", n_steps=8),
            daemon=True,
        )
        thread.start()
        try:
            time.sleep(0.3)
            rejected = hw.run_policy(policy_object=_TaggingPolicy("B"), instruction="b", n_steps=8)

            assert rejected["status"] == "error"
            # No rollout payload: it never ran, so there is nothing to report.
            assert not any("json" in block for block in rejected["content"]), rejected
        finally:
            thread.join(timeout=30.0)


class TestStartTaskExclusion:
    def test_a_second_start_task_is_not_queued(self):
        """The single-worker executor silently QUEUED the second rollout."""
        driver = _Bus(connect_seconds=0.5)
        hw = _robot(driver)
        try:
            first = hw.start_task("a", policy_port=1, duration=1.0)
            assert first["status"] == "success", first

            second = hw.start_task("b", policy_port=1, duration=1.0)

            assert second["status"] == "error", second
            assert "Task already running" in _text(second)
        finally:
            future = hw._task_state.task_future
            if future is not None:
                future.cancel()
            hw._executor.shutdown(wait=False)

    def test_run_policy_is_rejected_while_a_background_task_is_pending(self):
        """The two entry points must exclude EACH OTHER, not just themselves."""
        driver = _Bus(connect_seconds=0.5)
        hw = _robot(driver)
        try:
            assert hw.start_task("a", policy_port=1, duration=1.0)["status"] == "success"
            policy = _TaggingPolicy("B")

            rejected = hw.run_policy(policy_object=policy, instruction="b", n_steps=4)

            assert rejected["status"] == "error", rejected
            assert policy.inferences == 0
        finally:
            future = hw._task_state.task_future
            if future is not None:
                future.cancel()
            hw._executor.shutdown(wait=False)


class TestTheLockIsReleased:
    def test_a_completed_rollout_frees_the_slot(self):
        """Exclusivity must not become a one-shot robot."""
        driver = _Bus(connect_seconds=0.0)
        hw = _robot(driver)

        first = hw.run_policy(policy_object=_TaggingPolicy("A"), instruction="a", n_steps=4)
        second = hw.run_policy(policy_object=_TaggingPolicy("B"), instruction="b", n_steps=4)

        assert first["status"] == "success", first
        assert second["status"] == "success", second
        assert set(driver.writers) == {"A", "B"}

    def test_a_raising_rollout_frees_the_slot(self):
        """The release must be in a finally, or one crash bricks the robot."""
        driver = _Bus(connect_seconds=0.0)
        hw = _robot(driver)

        class _Exploding(_TaggingPolicy):
            async def get_actions(self, observation, instruction, **kwargs):
                raise RuntimeError("inference blew up")

        hw.run_policy(policy_object=_Exploding("X"), instruction="x", n_steps=4)

        assert not hw._task_lock.locked(), "the lock survived a failed rollout"
        recovered = hw.run_policy(policy_object=_TaggingPolicy("A"), instruction="a", n_steps=4)
        assert recovered["status"] == "success", recovered

    def test_the_slot_is_free_on_a_fresh_robot(self):
        hw = _robot(_Bus(connect_seconds=0.0))

        assert hw._task_busy_error("probe") is None
        assert not hw._task_lock.locked()


class TestGuardOnHalfBuiltInstances:
    def test_a_missing_lock_attribute_is_created_not_raised(self):
        """Robot.__new__ paths (tests, mesh shims) must not AttributeError."""
        hw = _robot(_Bus(connect_seconds=0.0))
        del hw._task_lock

        assert hw._task_busy_error("probe") is None
        assert isinstance(hw._task_lock, type(threading.Lock()))


class TestRejectionTextIsAscii:
    def test_busy_message_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        driver = _Bus(connect_seconds=1.0)
        hw = _robot(driver)
        thread = threading.Thread(
            target=lambda: hw.run_policy(policy_object=_TaggingPolicy("A"), instruction="a", n_steps=8),
            daemon=True,
        )
        thread.start()
        try:
            time.sleep(0.3)
            rejected = hw.run_policy(policy_object=_TaggingPolicy("B"), instruction="b", n_steps=8)

            assert _text(rejected).isascii()
        finally:
            thread.join(timeout=30.0)


class TestConcurrentBurst:
    def test_eight_simultaneous_callers_yield_exactly_one_rollout(self):
        """The lock, not the status flip, is what has to hold under a burst."""
        driver = _Bus(connect_seconds=0.3)
        hw = _robot(driver)
        policies = [_TaggingPolicy(f"P{index}") for index in range(8)]
        results: list[dict] = []
        results_lock = threading.Lock()
        gate = threading.Barrier(len(policies))

        def caller(policy: _TaggingPolicy) -> None:
            gate.wait()
            result = hw.run_policy(policy_object=policy, instruction=policy.who, n_steps=4)
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=caller, args=(policy,), daemon=True) for policy in policies]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)
        assert not any(thread.is_alive() for thread in threads)

        accepted = [result for result in results if result["status"] == "success"]
        assert len(accepted) == 1, [result["status"] for result in results]
        assert driver.max_concurrent == 1, f"{driver.max_concurrent} concurrent bus transactions"
        # Exactly one intent on the wire.
        assert len(set(driver.writers)) == 1, sorted(set(driver.writers))
