"""Regression for GH #4167: the real arm tool refuses an input field outside its schema.

``Robot.stream`` read its input with ``input_data.get(...)`` and never looked at
the keys it did not read, so ``execute(..., bogus=1)`` went on to the operator
gate and the arm as if the field had not been sent, while the sim tool refused
the same mistake by name (``Unknown parameter 'bogus' for action 'get_state'.
Valid: []``). A misspelt ``policy_config`` on a real arm was therefore a rollout
with the default configuration, approved by an operator who saw no misspelling.

Pinned here: the hardware tool refuses a key outside the action's own fields
BEFORE the gate, with the sentence the sim tool uses (one helper, so the two
cannot drift), and the fields it does document still pass.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from strands.types.tools import ToolUse

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.simulation.base import unknown_parameter_error
from tests._hardware_robot import hardware_robot_on


@pytest.fixture
def arm(monkeypatch: pytest.MonkeyPatch) -> tuple[HwRobot, list[str], MagicMock]:
    hw = hardware_robot_on(object(), tool_name="test_arm", control_frequency=30.0)
    reached: list[str] = []

    def _execute(instruction: str, port: Any, host: str, provider: str, duration: float, **kw: Any) -> dict[str, Any]:
        reached.append("execute")
        return {"status": "success", "content": [{"text": "done"}]}

    hw._execute_task_sync = _execute  # type: ignore[assignment]
    hw.start_task = _execute  # type: ignore[assignment]
    hw.stop_task = lambda: {"status": "success", "content": [{"text": "stopped"}]}  # type: ignore[assignment]
    hw.get_task_status = lambda: {"status": "success", "content": [{"text": "idle"}]}  # type: ignore[assignment]
    gate = MagicMock(name="gate_motion", return_value=None)
    monkeypatch.setattr(hw, "_gate_motion", gate)
    return hw, reached, gate


def _stream(hw: HwRobot, **inp: Any) -> dict[str, Any]:
    tool_use = cast(ToolUse, {"toolUseId": "tu", "input": inp})

    async def _run() -> list:
        return [ev async for ev in hw.stream(tool_use, {})]

    return asyncio.run(_run())[-1].tool_result


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


@pytest.mark.parametrize("action", ["execute", "start"])
def test_a_field_outside_the_schema_is_refused_before_the_gate(arm, action: str) -> None:
    hw, reached, gate = arm
    result = _stream(hw, action=action, instruction="pick", policy_provider="mock", bogus=1)
    assert result["status"] == "error", result
    text = _text(result)
    assert "Unknown parameter 'bogus'" in text and f"for action '{action}'" in text
    assert "policy_config" in text and "instruction" in text
    assert reached == [] and not gate.called


def test_the_refusal_is_the_sim_tools_sentence(arm) -> None:
    """One helper builds both tools' sentence, so the two cannot drift apart."""
    hw, _, _ = arm
    result = _stream(hw, action="stop", bogus=1)
    valid: list[str] = []
    assert _text(result) == _text(unknown_parameter_error(["bogus"], "stop", valid))


def test_a_close_misspelling_is_named(arm) -> None:
    hw, reached, _ = arm
    result = _stream(hw, action="execute", instruction="pick", policy_provider="mock", policy_confg={"x": 1})
    assert result["status"] == "error"
    assert "Did you mean" in _text(result) and "policy_config" in _text(result)
    assert reached == []


@pytest.mark.parametrize(
    ("action", "extra"),
    [
        ("execute", {"instruction": "pick", "policy_provider": "mock", "duration": 5, "policy_host": "localhost"}),
        ("start", {"instruction": "pick", "policy_provider": "mock", "policy_port": None}),
        ("status", {}),
        ("stop", {}),
    ],
)
def test_the_documented_fields_still_pass(arm, action: str, extra: dict[str, Any]) -> None:
    hw, _, _ = arm
    result = _stream(hw, action=action, **extra)
    assert result["status"] == "success", result


def test_an_unknown_action_is_still_answered_as_an_unknown_action(arm) -> None:
    hw, _, _ = arm
    result = _stream(hw, action="dance", bogus=1)
    assert result["status"] == "error"
    assert "dance" in _text(result) and "Unknown parameter" not in _text(result)


def test_every_schema_property_is_an_accepted_field(arm) -> None:
    """The tool's published schema and the per action field table name the same keys."""
    hw, _, _ = arm
    from strands_robots import hardware_robot

    schema_keys = set(hw.tool_spec["inputSchema"]["json"]["properties"]) - {"action"}
    table_keys = set().union(*hardware_robot.TOOL_FIELDS.values())
    assert schema_keys == table_keys
