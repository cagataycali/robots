"""A rollout envelope that echoes the instruction says when the policy never read it.

Measured with an agent driving ``Robot("so101", mode="real")`` and no policy
server: it chose ``mock`` (correct - the only in-process provider), the task
envelope came back ``Task: 'Wave the arm' - completed ... Policy: mock on
localhost:None``, and the agent told the user the arm had waved. Asked
plainly, a second run said the arm "stayed still" because mock "doesn't send
any real commands" - wrong the other way: ``MockPolicy`` drives every joint
in a sinusoid whatever the task says. The simulation's ``run_policy`` paired
the same two facts the same way: ``MockPolicy | wave the arm``.

``Policy.reads_instruction`` is the contract (default ``True``), and
:func:`instruction_not_read_notice` is the one sentence both envelopes append
when it is ``False``. The hardware envelope also stops describing an
in-process provider as running ``on localhost:None``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import TaskStatus
from strands_robots.policies.base import Policy, instruction_not_read_notice
from strands_robots.policies.mock import MockPolicy


class _Reader(Policy):
    """A policy that acts on the words - the default contract."""

    @property
    def provider_name(self) -> str:
        return "reader"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(self, observation_dict, instruction, **kwargs):  # type: ignore[override]
        return []


class TestTheContract:
    def test_the_default_is_that_a_policy_reads_its_instruction(self) -> None:
        assert _Reader().reads_instruction is True
        assert instruction_not_read_notice(_Reader()) is None

    def test_mock_declares_it_does_not(self) -> None:
        assert MockPolicy().reads_instruction is False

    def test_the_notice_names_the_policy_and_that_the_robot_was_commanded_anyway(self) -> None:
        notice = instruction_not_read_notice(MockPolicy())
        assert notice is not None
        assert notice.startswith("Note: MockPolicy does not read the instruction.")
        assert "were commanded to the robot" in notice
        assert "nothing above means the task was performed" in notice

    def test_an_object_without_the_property_is_read_as_a_reader(self) -> None:
        assert instruction_not_read_notice(object()) is None


@pytest.fixture
def hw() -> Iterator[HwRobot]:
    from strands_robots import Robot

    robot = Robot("so101", mode="real", port=os.devnull)
    yield robot
    robot.cleanup()


def _completed(hw: HwRobot, policy) -> None:
    async def rollout(*_a, **_k) -> None:
        hw._task_state.status = TaskStatus.COMPLETED
        hw._task_state.step_count = 7
        hw._task_state.duration = 0.3
        hw._task_state.policy = policy

    hw._execute_task_async = rollout  # type: ignore[method-assign]


class TestTheHardwareEnvelope:
    def test_a_mock_rollout_carries_the_notice(self, hw: HwRobot) -> None:
        _completed(hw, MockPolicy())
        text = hw._execute_task_sync("wave", None, "localhost", "mock", 0.3)["content"][0]["text"]
        assert "Task: 'wave' - completed" in text
        assert "Note: MockPolicy does not read the instruction." in text

    def test_an_in_process_provider_is_not_placed_on_a_server(self, hw: HwRobot) -> None:
        _completed(hw, MockPolicy())
        text = hw._execute_task_sync("wave", None, "localhost", "mock", 0.3)["content"][0]["text"]
        assert "Policy: mock (built in process, no server)" in text
        assert "None" not in text

    def test_a_server_provider_keeps_its_address_and_gets_no_notice(self, hw: HwRobot) -> None:
        _completed(hw, _Reader())
        text = hw._execute_task_sync("wave", 5555, "localhost", "groot", 0.3)["content"][0]["text"]
        assert "Policy: groot on localhost:5555" in text
        assert "Note:" not in text

    def test_a_pre_built_object_is_judged_by_its_own_contract(self, hw: HwRobot) -> None:
        mock = MockPolicy()
        _completed(hw, mock)
        text = hw._execute_task_sync("wave", None, "localhost", "mock", 0.3, policy_object=mock)["content"][0]["text"]
        assert "Policy: MockPolicy (pre-built object)" in text
        assert "Note: MockPolicy does not read the instruction." in text

    def test_the_driven_policy_is_cleared_when_the_next_task_begins(self, hw: HwRobot) -> None:
        """The real rollout resets ``policy`` with the rest of the state before it connects."""
        hw._task_state.policy = MockPolicy()

        async def refuse_to_connect() -> tuple[bool, str]:
            return False, "no bus in this test"

        hw._connect_robot = refuse_to_connect  # type: ignore[method-assign]
        result = hw._execute_task_sync("wave", 5555, "localhost", "groot", 0.1)
        assert result["status"] == "error"
        assert hw._task_state.policy is None
        assert "Note:" not in result["content"][0]["text"]


@pytest.mark.skipif(os.environ.get("MUJOCO_GL", "") == "disabled", reason="needs a MuJoCo context")
class TestTheSimulationEnvelope:
    def test_run_policy_with_mock_says_the_instruction_was_not_read(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
        from strands_robots import Robot

        arm = Robot("so101", mode="sim")
        try:
            r = arm(action="run_policy", robot_name="so101", policy_provider="mock", duration=0.2, instruction="wave")
        finally:
            arm.destroy()
        assert r["status"] == "success"
        text = next(c["text"] for c in r["content"] if "text" in c)
        payload = next(c["json"] for c in r["content"] if "json" in c)
        assert "MockPolicy | wave" in text
        assert "Note: MockPolicy does not read the instruction." in text
        assert payload["instruction_read"] is False
