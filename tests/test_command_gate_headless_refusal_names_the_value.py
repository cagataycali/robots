"""The headless refusal names the allowlist VALUE, and says when the variable is already set.

Measured on a real SO-101 tool with ``STRANDS_ROBOT_COMMAND_ALLOW=so101`` (the
tool name; this gate matches the action): ``execute`` was refused with "Set
STRANDS_ROBOT_COMMAND_ALLOW or BYPASS_TOOL_CONSENT=true" - the variable the
caller had just set, with no word on what value would have counted.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import TaskStatus


def _text(result) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    robot = Robot("so101", mode="real", port=os.devnull)

    async def rollout(instruction, *args, **kwargs):
        robot._task_state.status = TaskStatus.COMPLETED

    robot._execute_task_async = rollout
    yield robot
    robot.cleanup()


def _execute(arm):
    async def call():
        last = None
        tool_use = {
            "toolUseId": "t",
            "name": "so101",
            "input": {"action": "execute", "instruction": "wave", "policy_provider": "mock", "duration": 1},
        }
        async for ev in arm.stream(tool_use, {}):
            last = ev
        return last.tool_result

    return asyncio.run(call())


class TestHeadlessRefusal:
    def test_unset_variable_names_the_value_to_set(self, arm) -> None:
        text = _text(_execute(arm))
        assert "Set STRANDS_ROBOT_COMMAND_ALLOW=execute (or STRANDS_ROBOT_COMMAND_ALLOW=*)" in text
        assert "is set but does not name this command" not in text

    def test_a_set_but_non_matching_variable_is_named_as_such(self, arm, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "so101")
        text = _text(_execute(arm))
        assert "STRANDS_ROBOT_COMMAND_ALLOW='so101' is set but does not name this command." in text
        assert "Set STRANDS_ROBOT_COMMAND_ALLOW=execute" in text

    def test_a_matching_variable_is_not_refused(self, arm, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "execute")
        result = _execute(arm)
        assert "No tool_context available" not in _text(result)
