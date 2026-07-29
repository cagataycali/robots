"""A stop pressed while the robot is still connecting must actually stop it.

``strands_robots.hardware_robot.Robot.stop_task`` is the interrupt an operator
reaches for, and the fleet ``{"action": "stop"}`` dispatch routes into it. But
``_execute_task_async`` spends the whole hardware bring-up in
``TaskStatus.CONNECTING`` - a motors-bus handshake plus ``warmup_s`` per camera,
seconds on a real arm - and the guard only recognised ``RUNNING``. So a stop in
that window was answered ``status="success"`` with "No task running to stop", and
the arm then moved anyway once the bring-up finished.

Signalling through ``_task_state.status`` alone cannot express the request:
``_execute_task_async`` writes ``RUNNING`` once bring-up completes, overwriting
any ``STOPPED`` recorded before it. These tests pin the latch-based contract:

    - a stop during connect, or during the policy build, commands the arm zero
      times and leaves the task ``STOPPED``;
    - the refusal happens before the policy is initialized;
    - the request is latched even when the status says there is nothing to stop,
      so a stop landing in the gap before the ``RUNNING`` write is not lost and
      the task does not report itself completed;
    - the latch is cleared by the next task, so a stop never leaks forward;
    - a stop mid-rollout, and a stop on an idle or terminal robot, behave as
      before.

No serial port is opened and no arm is commanded: the lerobot driver is an
in-memory fake and each bring-up stage is an event the test opens explicitly, so
nothing here depends on wall-clock timing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies import Policy

# Generous ceiling for every bounded wait: each is satisfied by an event the
# test itself sets, so reaching it means the contract is broken, not that the
# host was slow.
DEADLINE = 10.0


class _FakeArm:
    """In-memory stand-in for a connected lerobot robot."""

    def __init__(self) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": {}})()

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


class _OneStepPolicy(Policy):
    """Emits a single action per query so the loop keeps re-querying."""

    @property
    def provider_name(self) -> str:
        return "test"

    def set_robot_state_keys(self, keys: list[str]) -> None:
        return None

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.1}]


class _Rig:
    """A ``Robot`` whose bring-up stages are gates the test opens explicitly."""

    def __init__(self) -> None:
        self.arm = _FakeArm()
        self.connect_gate = threading.Event()
        self.policy_gate = threading.Event()
        self.policy_initialized: list[str] = []
        self.result: dict[str, Any] | None = None

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "test_arm"
        hw.action_horizon = 1
        hw.data_config = None
        hw.control_frequency = 500.0
        hw.action_sleep_time = 1.0 / 500.0
        hw._task_state = RobotTaskState()
        hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
        hw._shutdown_event = threading.Event()
        hw._stop_requested = threading.Event()
        hw._task_admission = threading.Lock()
        hw._task_claimed = False
        hw.mesh = None
        hw.peer_id = None
        hw.robot = self.arm

        async def connect() -> tuple[bool, str]:
            await asyncio.to_thread(self.connect_gate.wait, DEADLINE)
            return True, ""

        async def initialize_policy(policy: Any) -> bool:
            self.policy_initialized.append(type(policy).__name__)
            await asyncio.to_thread(self.policy_gate.wait, DEADLINE)
            return True

        def publish(observation: dict[str, Any], skip_images: bool = False) -> None:
            return None

        hw._connect_robot = connect  # type: ignore[method-assign]
        hw._initialize_policy = initialize_policy  # type: ignore[method-assign]
        hw._publish_ros_telemetry = publish  # type: ignore[method-assign]
        self.robot = hw

    def start(self, *, duration: float = 5.0, n_steps: int | None = 3) -> threading.Thread:
        """Drive a rollout on its own thread, recording its returned payload."""

        def target() -> None:
            self.result = self.robot.run_policy(_OneStepPolicy(), "pick the cube", duration=duration, n_steps=n_steps)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def wait_for(self, predicate: Any, what: str) -> None:
        """Spin until ``predicate`` holds, failing the test if it never does."""
        deadline = time.monotonic() + DEADLINE
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        pytest.fail(f"timed out waiting for {what} (status={self.robot._task_state.status.value})")

    def open_all_gates(self) -> None:
        self.connect_gate.set()
        self.policy_gate.set()

    def finish(self, thread: threading.Thread) -> None:
        self.open_all_gates()
        thread.join(timeout=DEADLINE)
        assert not thread.is_alive(), "rollout thread did not finish"
        self.robot._executor.shutdown(wait=False)


@pytest.fixture
def rig() -> Any:
    return _Rig()


class TestStopWhileConnecting:
    def test_the_arm_is_never_commanded(self, rig):
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")

        rig.robot.stop_task()
        rig.finish(thread)

        assert rig.arm.sent_actions == []
        assert rig.robot._task_state.status == TaskStatus.STOPPED

    def test_the_stop_is_reported_as_a_stop(self, rig):
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")

        result = rig.robot.stop_task()
        rig.finish(thread)

        text = result["content"][0]["text"]
        assert result["status"] == "success"
        assert "No task running to stop" not in text
        assert "Task stopped (during connect)" in text
        assert "pick the cube" in text

    def test_the_policy_is_never_initialized(self, rig):
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")

        rig.robot.stop_task()
        rig.finish(thread)

        assert rig.policy_initialized == []

    def test_the_blocking_call_reports_the_stop_not_a_completed_task(self, rig):
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")

        rig.robot.stop_task()
        rig.finish(thread)

        assert rig.result is not None
        assert "stopped" in rig.result["content"][0]["text"]
        assert "completed" not in rig.result["content"][0]["text"]

    def test_no_stale_duration_is_reported(self, rig):
        rig.robot._task_state.duration = 99.0  # left by an earlier task
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")

        result = rig.robot.stop_task()
        rig.finish(thread)

        assert "Duration: 0.0s" in result["content"][0]["text"]
        assert "99.0" not in result["content"][0]["text"]


class TestStopWhileThePolicyIsBuilt:
    def test_the_arm_is_never_commanded(self, rig):
        thread = rig.start()
        rig.connect_gate.set()
        rig.wait_for(lambda: rig.policy_initialized, "the policy build stage")

        rig.robot.stop_task()
        rig.finish(thread)

        assert rig.arm.sent_actions == []
        assert rig.robot._task_state.status == TaskStatus.STOPPED


class TestTheRequestIsLatched:
    def test_a_stop_is_latched_even_when_nothing_is_running(self, rig):
        result = rig.robot.stop_task()

        assert result["status"] == "success"
        assert "No task running to stop" in result["content"][0]["text"]
        # The status reply is unchanged, but the request itself is recorded: a
        # stop that races the rollout's own status writes must not be lost.
        assert rig.robot._stop_requested.is_set()

    def test_a_stop_in_the_gap_before_running_does_not_report_completed(self, rig):
        # The gap a status-only signal loses: after the rollout's final stage
        # check and before it writes RUNNING over the STOPPED status. Emulated
        # by latching immediately after that last check returns "not stopped".
        real = rig.robot._honor_stop_request
        verdicts: list[bool] = []

        def spy() -> bool:
            verdict = real()
            verdicts.append(verdict)
            if len(verdicts) == 2:
                rig.robot._stop_requested.set()
            return verdict

        rig.robot._honor_stop_request = spy
        thread = rig.start()
        rig.finish(thread)

        assert verdicts == [False, False], "both stage checks should have passed"
        assert rig.arm.sent_actions == []
        assert rig.robot._task_state.status == TaskStatus.STOPPED


class TestTheLatchDoesNotLeakForward:
    def test_the_next_task_runs_after_a_stop_on_an_idle_robot(self, rig):
        rig.robot.stop_task()
        assert rig.robot._stop_requested.is_set()

        rig.open_all_gates()
        thread = rig.start()
        rig.finish(thread)

        assert len(rig.arm.sent_actions) == 3
        assert rig.robot._task_state.status == TaskStatus.COMPLETED

    def test_a_task_stopped_during_connect_does_not_pre_empt_the_next_one(self, rig):
        first = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")
        rig.robot.stop_task()
        rig.open_all_gates()
        first.join(timeout=DEADLINE)
        assert rig.arm.sent_actions == []

        second = rig.start()
        rig.finish(second)

        assert len(rig.arm.sent_actions) == 3
        assert rig.robot._task_state.status == TaskStatus.COMPLETED


class TestUnchangedStopBehaviour:
    def test_a_stop_mid_rollout_still_stops_it(self, rig):
        rig.open_all_gates()
        thread = rig.start(duration=30.0, n_steps=None)
        rig.wait_for(lambda: rig.arm.sent_actions, "the first commanded action")

        result = rig.robot.stop_task()
        thread.join(timeout=DEADLINE)
        rig.robot._executor.shutdown(wait=False)

        assert result["status"] == "success"
        assert "Task stopped:" in result["content"][0]["text"]
        assert "(during connect)" not in result["content"][0]["text"]
        assert rig.robot._task_state.status == TaskStatus.STOPPED

    @pytest.mark.parametrize(
        "state",
        [TaskStatus.IDLE, TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.ERROR],
    )
    def test_a_robot_with_no_live_task_reports_nothing_to_stop(self, rig, state):
        rig.robot._task_state.status = state

        result = rig.robot.stop_task()

        assert result["status"] == "success"
        assert f"No task running to stop (current: {state.value})" in result["content"][0]["text"]

    def test_the_messages_are_plain_ascii(self, rig):
        thread = rig.start()
        rig.wait_for(lambda: rig.robot._task_state.status == TaskStatus.CONNECTING, "the connect stage")
        stopped = rig.robot.stop_task()["content"][0]["text"]
        rig.finish(thread)
        idle = rig.robot.stop_task()["content"][0]["text"]

        for text in (stopped, idle):
            assert text.isascii(), text
