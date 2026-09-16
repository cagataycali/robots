"""A bring-up that fails after the arm connected disconnects it again (torque released).

lerobot's ``connect()`` ends in ``configure()``, whose ``torque_disabled()``
block re-enables torque on exit: the arm goes stiff where it stands the moment
it connects, before any policy exists. Measured on the SO-101 tool with a
policy that could not be built: task ERROR, ``robot.is_connected`` True, torque
on for the rest of the session with nothing to drive it. A connection the
caller made before the task is theirs and is left alone; a rollout that fails
while RUNNING is not released (an arm mid-motion dropping under gravity is the
hazard, holding its pose is not).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import TaskStatus


class _Spy:
    """Stand-in for the lerobot driver's connection surface, recording the order of calls."""

    def __init__(self, robot, monkeypatch) -> None:
        self.log: list[str] = []
        self.connected = False
        monkeypatch.setattr(robot, "connect", self._connect)
        monkeypatch.setattr(robot, "disconnect", self._disconnect)
        # Class-level properties, restored by monkeypatch so no other module
        # inherits a driver class that answers from this spy.
        monkeypatch.setattr(type(robot), "is_connected", property(lambda _s: self.connected))
        monkeypatch.setattr(type(robot), "is_calibrated", property(lambda _s: True))

    def _connect(self, calibrate: bool = True) -> None:
        self.log.append("connect")
        self.connected = True

    def _disconnect(self) -> None:
        self.log.append("disconnect")
        self.connected = False


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
    robot = Robot("so101", mode="real", port=os.devnull)
    spy = _Spy(robot.robot, monkeypatch)
    yield robot, spy
    robot.cleanup()


async def _execute(arm, **extra):
    last = None
    inp = {"action": "execute", "instruction": "wave", "policy_provider": "mock", "duration": 1, **extra}
    async for ev in arm.stream({"toolUseId": "t", "name": "so101", "input": inp}, {}):
        last = ev
    return last.tool_result


def _text(result) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


class TestPolicyCannotBeBuilt:
    def test_the_arm_this_task_connected_is_released(self, arm) -> None:
        robot, spy = arm

        async def boom(*args, **kwargs):
            raise RuntimeError("checkpoint 'nobody/none' not found")

        robot._get_policy = boom
        text = _text(asyncio.run(_execute(robot)))
        assert robot._task_state.status is TaskStatus.ERROR
        assert "checkpoint 'nobody/none' not found" in text
        assert "disconnected again (torque released)" in text
        assert spy.log == ["connect", "disconnect"]
        assert not spy.connected

    def test_a_connection_the_caller_made_first_is_left_alone(self, arm) -> None:
        robot, spy = arm
        spy._connect()  # the caller connected before the task

        async def boom(*args, **kwargs):
            raise RuntimeError("server down")

        robot._get_policy = boom
        text = _text(asyncio.run(_execute(robot)))
        assert "server down" in text
        assert "disconnected again" not in text
        assert spy.log == ["connect"]
        assert spy.connected


class TestPolicyCannotBeInitialized:
    def test_the_arm_is_released_and_the_reason_kept(self, arm) -> None:
        robot, spy = arm

        async def no_init(policy):
            return False

        robot._initialize_policy = no_init
        text = _text(asyncio.run(_execute(robot)))
        assert "Failed to initialize policy" in text
        assert "disconnected again (torque released)" in text
        assert spy.log == ["connect", "disconnect"]


class TestARolloutThatFailsWhileRunningHoldsItsPose:
    def test_no_release_after_running(self, arm) -> None:
        robot, spy = arm

        async def rollout(instruction, *args, **kwargs):
            await robot._connect_robot()
            robot._task_state.status = TaskStatus.RUNNING
            raise RuntimeError("policy step failed")

        robot._execute_task_async = rollout
        asyncio.run(_execute(robot))
        assert spy.log == ["connect"]
        assert spy.connected


class TestDisconnectThatFails:
    def test_the_policy_error_stays_the_headline(self, arm) -> None:
        robot, spy = arm

        def stuck() -> None:
            raise OSError("bus busy")

        robot.robot.disconnect = stuck  # instance attr; the fixture's robot is discarded after the test

        async def boom(*args, **kwargs):
            raise RuntimeError("no checkpoint")

        robot._get_policy = boom
        text = _text(asyncio.run(_execute(robot)))
        assert text.index("no checkpoint") < text.index("disconnecting it failed: bus busy")
        assert "still connected (torque on)" in text
