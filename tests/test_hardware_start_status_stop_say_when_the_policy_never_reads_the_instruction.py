"""``start`` / ``status`` / ``stop`` on the real robot carry the instruction-not-read notice.

The ``execute`` envelope gained the sentence; the background lifecycle did not.
Measured with an agent: ``start`` "trace a small circle" with ``mock`` →
``Task started: 'trace a small circle'``, ``status`` → ``RUNNING … Steps: 18``,
``stop`` → ``Task stopped … Steps completed: 34``, and the agent summarised
"Task launched successfully … RUNNING — 18 steps completed … stopped cleanly".
Nothing in three answers said the circle was never attempted.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from concurrent.futures import Future

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import TaskStatus
from strands_robots.policies.base import Policy, instruction_not_read_notice, provider_policy_class
from strands_robots.policies.mock import MockPolicy
from tests.test_hardware_control_loop_rate_guard import _FakeArm


@pytest.fixture
def arm() -> Iterator[HwRobot]:
    robot = Robot("so101", mode="real", port=os.devnull)
    yield robot
    robot.cleanup()


def _text(result: dict) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


class TestTheClassAnswers:
    def test_reads_instruction_is_readable_off_the_class(self) -> None:
        assert MockPolicy.reads_instruction is False
        assert Policy.reads_instruction is True

    def test_registry_resolves_mock_to_its_class(self) -> None:
        assert provider_policy_class("mock") is MockPolicy
        assert provider_policy_class("test") is MockPolicy  # shorthand
        assert provider_policy_class("no-such-provider") is None
        assert provider_policy_class(None) is None

    def test_notice_from_a_class_in_the_present_tense(self) -> None:
        text = instruction_not_read_notice(MockPolicy, pending=True)
        assert text is not None
        assert text.startswith("Note: MockPolicy does not read the instruction.")
        assert "are commanded" in text and "will mean the task was performed" in text

    def test_notice_from_an_instance_in_the_past_tense(self) -> None:
        text = instruction_not_read_notice(MockPolicy())
        assert text is not None and "were commanded" in text


class TestStart:
    def test_start_with_mock_names_it_before_the_build(self, arm, monkeypatch) -> None:
        submitted: list = []

        class _Executor:
            def submit(self, fn, *a, **k):
                submitted.append((fn, a, k))
                return Future()

        monkeypatch.setattr(arm, "_executor", _Executor())
        result = arm.start_task("trace a small circle", policy_provider="mock")
        assert result["status"] == "success", _text(result)
        text = _text(result)
        assert "Task started: 'trace a small circle'" in text
        assert "Note: MockPolicy does not read the instruction." in text
        assert text.index("does not read") < text.index("Use action='status'")
        assert submitted

    def test_start_with_a_reading_provider_carries_no_notice(self, arm, monkeypatch) -> None:
        class _Executor:
            def submit(self, fn, *a, **k):
                return Future()

        monkeypatch.setattr(arm, "_executor", _Executor())
        result = arm.start_task("trace a small circle", policy_port=5555, policy_provider="groot")
        assert result["status"] == "success", _text(result)
        assert "does not read" not in _text(result)


class TestStatusAndStop:
    def _running_mock(self, arm) -> None:
        st = arm._task_state
        st.status = TaskStatus.RUNNING
        st.instruction = "trace a small circle"
        st.start_mono = time.monotonic()
        st.step_count = 18
        st.policy = MockPolicy()

    def test_status_running_says_so_in_the_present_tense(self, arm) -> None:
        self._running_mock(arm)
        text = _text(arm.get_task_status())
        assert "Robot Status: RUNNING" in text and "Steps: 18" in text
        assert "Note: MockPolicy does not read the instruction." in text
        assert "are commanded" in text

    def test_status_completed_says_so_in_the_past_tense(self, arm) -> None:
        self._running_mock(arm)
        arm._task_state.status = TaskStatus.COMPLETED
        text = _text(arm.get_task_status())
        assert "Total Steps: 18" in text and "were commanded" in text

    def test_status_idle_carries_no_notice(self, arm) -> None:
        assert "does not read" not in _text(arm.get_task_status())

    def test_stop_carries_the_notice(self, arm) -> None:
        self._running_mock(arm)
        text = _text(arm.stop_task())
        assert "Task stopped" in text and "Steps completed: 18" in text
        assert "Note: MockPolicy does not read the instruction." in text
        assert "were commanded" in text

    def test_status_with_a_reading_policy_carries_no_notice(self, arm) -> None:
        self._running_mock(arm)

        class _Reads(MockPolicy):
            reads_instruction = True

        arm._task_state.policy = _Reads()
        assert "does not read" not in _text(arm.get_task_status())


class TestTheRolloutItselfRecordsThePolicyItDrove:
    """Driven by the real rollout, with nothing about the policy set by hand.

    Every cell above states the task machine directly, so the one line where
    ``_execute_task_async`` records the policy it built is unpinned by them:
    delete it and they all stay green while ``start`` / ``status`` / ``stop``
    go silent again on a real arm. This drives the real loop on an in-memory
    arm instead - the notice has to survive the whole path.
    """

    @staticmethod
    def _in_memory(arm, monkeypatch) -> None:
        arm.robot = _FakeArm()

        async def connected() -> tuple[bool, str]:
            return True, ""

        async def ready() -> bool:
            return True

        monkeypatch.setattr(arm, "_connect_robot", connected)
        monkeypatch.setattr(arm, "_initialize_policy", lambda policy: ready())
        monkeypatch.setattr(arm, "_publish_ros_telemetry", lambda *a, **k: None)

    @staticmethod
    def _wait_until_stepping(arm) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if arm._task_state.status is TaskStatus.RUNNING and arm._task_state.step_count > 0:
                return
            if arm._task_state.status in (TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.STOPPED):
                break
            time.sleep(0.01)
        raise AssertionError(f"rollout never stepped: {arm._task_state}")

    def test_all_three_envelopes_carry_it_with_no_task_state_set_by_hand(self, arm, monkeypatch) -> None:
        self._in_memory(arm, monkeypatch)
        started = _text(arm.start_task("trace a small circle", policy_provider="mock", duration=10.0))
        self._wait_until_stepping(arm)
        running = _text(arm.get_task_status())
        stopped = _text(arm.stop_task())

        assert isinstance(arm._task_state.policy, MockPolicy)  # recorded by the loop, not the test
        assert "Note: MockPolicy does not read the instruction." in started
        assert "Note: MockPolicy does not read the instruction." in running
        assert "Note: MockPolicy does not read the instruction." in stopped
        assert "are commanded" in running and "were commanded" in stopped
        assert f"Steps: {arm._task_state.step_count}" in running

    def test_status_still_says_it_when_the_device_cannot_be_read(self, arm, monkeypatch) -> None:
        """The notice sits with the task state, so the device-facts failure path keeps it."""
        self._in_memory(arm, monkeypatch)
        arm.start_task("trace a small circle", policy_provider="mock", duration=10.0)
        self._wait_until_stepping(arm)

        def unreadable() -> dict:
            raise OSError("no bus in this test")

        monkeypatch.setattr(arm, "_device_facts", unreadable)
        text = _text(arm.get_task_status())
        assert "Device:" not in text  # the facts really were unavailable
        assert "Note: MockPolicy does not read the instruction." in text
        arm.stop_task()
