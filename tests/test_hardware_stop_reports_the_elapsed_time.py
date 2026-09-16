"""``stop`` reports how long the task ran, and ``status`` afterwards agrees.

Measured on an SO-101 tool with a rollout stopped from outside after ~1.2 s:
``Task stopped: 'wave' / Duration: 0.0s / Steps completed: 24`` and then
``Robot Status: STOPPED … Total Duration: 0.0s / Total Steps: 24`` for good.
The rollout writes ``duration`` only when its own loop ends and ``status``
only while RUNNING, so a stop from outside froze whatever was last written.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import TaskStatus


def _text(result) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
    robot = Robot("so101", mode="real", port=os.devnull)

    async def rollout(instruction, *args, duration=30.0, **kwargs):
        state = robot._task_state
        state.status = TaskStatus.RUNNING
        state.start_mono = time.monotonic()
        while time.monotonic() - state.start_mono < duration:
            state.step_count += 1
            await asyncio.sleep(0.02)

    robot._execute_task_async = rollout
    yield robot
    robot.cleanup()


async def _call(arm, **inp):
    last = None
    async for ev in arm.stream({"toolUseId": "t", "name": "so101", "input": inp}, {}):
        last = ev
    return last.tool_result


class TestStopFromOutside:
    def test_stop_reports_the_elapsed_time_not_zero(self, arm) -> None:
        async def scenario():
            await _call(arm, action="start", instruction="wave", policy_provider="mock", duration=20)
            await asyncio.sleep(0.4)
            return await _call(arm, action="stop")

        text = _text(asyncio.run(scenario()))
        assert "Task stopped: 'wave'" in text
        assert "Duration: 0.0s" not in text
        assert arm._task_state.duration >= 0.3
        assert "Steps completed:" in text

    def test_status_afterwards_agrees_with_stop(self, arm) -> None:
        async def scenario():
            await _call(arm, action="start", instruction="wave", policy_provider="mock", duration=20)
            await asyncio.sleep(0.4)
            await _call(arm, action="stop")
            return await _call(arm, action="status")

        text = _text(asyncio.run(scenario()))
        assert "Robot Status: STOPPED" in text
        assert "Total Duration: 0.0s" not in text
        assert f"Total Duration: {arm._task_state.duration:.1f}s" in text

    def test_stop_during_connect_still_settles_a_duration(self, arm) -> None:
        arm._task_state.status = TaskStatus.CONNECTING
        arm._task_state.start_mono = time.monotonic() - 0.5
        text = _text(arm.stop_task())
        assert "(during connect)" in text
        assert arm._task_state.duration >= 0.5

    def test_stop_with_no_task_is_unchanged(self, arm) -> None:
        text = _text(arm.stop_task())
        assert text.startswith("No task running to stop")
