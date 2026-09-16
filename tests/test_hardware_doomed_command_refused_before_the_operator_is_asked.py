"""A real-robot ``execute``/``start`` the dispatcher would refuse on its inputs is refused before the operator is asked.

``Robot(mode="real")`` asks the operator to approve every ``execute`` and
``start``; the dispatcher then checks the inputs. Before this, a call with
``duration=-5`` or ``policy_port=99999`` raised the approval interrupt,
and the operator who typed "y" was answered with ``duration must be > 0``
- an approval spent on a command that could never have moved the arm, and
a second approval round for the agent's corrected retry. Every check here
is a pure function of the call and of the robot's own shut-down flag, so
it runs first; the dispatcher still runs it again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from strands import Agent
from strands.models.model import Model
from strands.types.tools import ToolUse

# The factory is how an agent builds a real robot; it is a function, so the
# class it returns is imported too for the annotations below.
from strands_robots import Robot
from strands_robots.hardware_robot import COMMAND_ALLOW_ENV
from strands_robots.hardware_robot import Robot as HwRobot


class _Silent(Model):
    def update_config(self, **kwargs: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def stream(self, *args: Any, **kwargs: Any):
        if False:
            yield

    async def structured_output(self, *args: Any, **kwargs: Any):
        if False:
            yield


def _events(robot: HwRobot, tool_input: dict[str, Any]) -> list[str]:
    """Return one tag per streamed event: ``result:<text>`` or ``interrupt``."""
    agent = Agent(model=_Silent(), tools=[], callback_handler=None)
    tool_use: ToolUse = {"toolUseId": "t1", "name": robot.tool_name, "input": tool_input}

    async def collect() -> list[str]:
        out = []
        async for ev in robot.stream(tool_use, {"agent": agent}):
            tr = getattr(ev, "tool_result", None)
            if tr is not None:
                out.append(f"{tr['status']}:{tr['content'][0]['text']}")
            else:
                out.append("interrupt")
        return out

    return asyncio.run(collect())


@pytest.fixture
def real(monkeypatch: pytest.MonkeyPatch) -> Iterator[HwRobot]:
    """A real-mode robot in a clean gate environment: nothing pre-approves the call.

    Both env vars decide whether the operator is asked at all, so both are
    cleared: with either one set the two "still asks the operator" cells below
    would pass for the wrong reason.
    """
    for name in ("BYPASS_TOOL_CONSENT", COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    robot = Robot("so101", mode="real", port="/dev/cu.does-not-exist")
    yield robot
    robot.cleanup()


DOOMED = [
    pytest.param(
        {"action": "execute", "instruction": "wave", "policy_provider": "mock", "duration": -5},
        "duration must be > 0",
        id="execute-negative-duration",
    ),
    pytest.param(
        {"action": "start", "instruction": "wave", "policy_provider": "mock", "duration": "ten"},
        "duration must be > 0",
        id="start-duration-not-a-number",
    ),
    pytest.param(
        {"action": "execute", "instruction": "wave", "policy_provider": "groot", "policy_port": 99999, "duration": 5},
        "invalid policy_port: 99999",
        id="execute-port-out-of-range",
    ),
    pytest.param(
        {"action": "start", "instruction": "wave", "policy_provider": "groot", "duration": 5},
        "policy_port is required",
        id="start-port-missing-for-groot",
    ),
]


@pytest.mark.parametrize("tool_input,expected", DOOMED)
def test_a_doomed_command_is_refused_without_asking_the_operator(real, tool_input, expected):
    events = _events(real, tool_input)
    assert len(events) == 1, events
    assert events[0].startswith("error:"), events
    assert expected in events[0]
    assert "interrupt" not in events


@pytest.mark.parametrize("tool_input,expected", DOOMED)
def test_the_refusal_is_the_dispatchers_own_wording(real, tool_input, expected):
    """Pre-gate and dispatcher speak with one voice: the method name leads."""
    events = _events(real, tool_input)
    method = "execute_task" if tool_input["action"] == "execute" else "start_task"
    assert events[0].startswith(f"error:{method}: ")


@pytest.mark.parametrize("action", ["execute", "start"])
def test_a_sound_command_still_asks_the_operator(real, action):
    events = _events(real, {"action": action, "instruction": "wave", "policy_provider": "mock", "duration": 5})
    assert events == ["interrupt"], events


def test_a_shut_down_robot_is_refused_before_the_operator_is_asked(real):
    real.cleanup()
    events = _events(real, {"action": "execute", "instruction": "wave", "policy_provider": "mock", "duration": 5})
    assert len(events) == 1 and events[0].startswith("error:"), events
    assert "interrupt" not in events


def test_the_dispatcher_still_checks_when_the_gate_is_bypassed(real, monkeypatch):
    """Defence in depth: the allowlist skips the operator, not the input checks."""
    monkeypatch.setenv(COMMAND_ALLOW_ENV, "*")
    events = _events(real, {"action": "execute", "instruction": "wave", "policy_provider": "mock", "duration": -5})
    assert events == ["error:execute_task: duration must be > 0, got -5."], events
