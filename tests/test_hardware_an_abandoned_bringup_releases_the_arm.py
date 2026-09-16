"""A bring-up that ends after the arm connected disconnects it again (torque released).

lerobot's ``connect()`` ends in ``configure()``, whose ``torque_disabled()``
block re-enables torque on exit: the arm goes stiff where it stands the moment
it connects, before any policy exists. Measured on the SO-101 tool with a policy
that could not be built: task ERROR, ``robot.is_connected`` True, torque on for
the rest of the session with nothing to drive it. The same held for a bring-up
an operator STOPPED, which is the likelier way to reach it.

A connection the caller made before the task is theirs and is left alone; a
rollout that fails while RUNNING is not released (an arm mid-motion dropping
under gravity is the hazard, holding its pose is not).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import TaskStatus

_RELEASED = "The robot was disconnected again (torque released) since nothing will drive it."


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


def _stop_during(robot, stage: str) -> None:
    """Latch a stop inside one of the two bring-up windows, as an operator would.

    ``connect`` is the motors-bus handshake plus per-camera warmup, ``policy``
    the policy build and initialize that follows it. Each is seconds long on a
    real arm, and each ends at a stage gate that abandons the rollout.
    """
    if stage == "connect":
        original = robot._connect_robot

        async def connect_then_stop():
            outcome = await original()
            robot._stop_requested.set()
            return outcome

        robot._connect_robot = connect_then_stop
    else:

        async def init_then_stop(policy):
            robot._stop_requested.set()
            return True  # the policy is ready; the stop is what ends the task

        robot._initialize_policy = init_then_stop


class TestPolicyCannotBeBuilt:
    def test_the_arm_this_task_connected_is_released(self, arm) -> None:
        robot, spy = arm

        async def boom(*args, **kwargs):
            raise RuntimeError("checkpoint 'nobody/none' not found")

        robot._get_policy = boom
        text = _text(asyncio.run(_execute(robot)))
        assert robot._task_state.status is TaskStatus.ERROR
        # The cause stays the headline, the release follows it, and a task that
        # really did fail is still labelled an error.
        assert f"Error: checkpoint 'nobody/none' not found {_RELEASED}" in text
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
        assert f"Failed to initialize policy {_RELEASED}" in text
        assert spy.log == ["connect", "disconnect"]


class TestStopDuringBringUp:
    """A stop abandons the bring-up, so the arm it energized has nothing to drive it."""

    @pytest.mark.parametrize("stage", ["connect", "policy"])
    def test_the_arm_this_task_connected_is_released(self, arm, stage) -> None:
        robot, spy = arm
        _stop_during(robot, stage)
        text = _text(asyncio.run(_execute(robot)))
        assert robot._task_state.status is TaskStatus.STOPPED
        assert spy.log == ["connect", "disconnect"]
        assert not spy.connected
        # A handled interrupt is not a failure, so the note is not labelled one.
        assert f"Note: {_RELEASED}" in text
        assert "Error:" not in text

    def test_a_connection_the_caller_made_first_is_left_alone(self, arm) -> None:
        robot, spy = arm
        spy._connect()  # the caller connected before the task
        _stop_during(robot, "connect")
        text = _text(asyncio.run(_execute(robot)))
        assert robot._task_state.status is TaskStatus.STOPPED
        assert spy.log == ["connect"]
        assert spy.connected
        assert "disconnected again" not in text

    def test_the_task_status_reports_the_release_as_a_note(self, arm) -> None:
        robot, spy = arm
        _stop_during(robot, "connect")
        asyncio.run(_execute(robot))
        status = _text(robot.get_task_status())
        assert f"Note: {_RELEASED}" in status
        assert "Error:" not in status


class TestARolloutThatFailsWhileRunningHoldsItsPose:
    def test_no_release_after_running(self, arm, monkeypatch) -> None:
        robot, spy = arm

        class _Stepless:
            """A policy whose first inference fails, once the rollout is RUNNING."""

            def set_robot_state_keys(self, keys) -> None:
                pass

            def set_control_frequency(self, frequency) -> None:
                pass

            def set_rtc_observed_delay(self, steps) -> None:
                pass

            def reset(self) -> None:
                pass

            async def get_actions(self, observation, instruction):
                assert robot._task_state.status is TaskStatus.RUNNING
                raise RuntimeError("policy step failed")

        monkeypatch.setattr("strands_robots.hardware_robot.read_observation", lambda _robot: {"shoulder.pos": 0.0})
        asyncio.run(robot._execute_task_async("wave", policy_object=_Stepless(), duration=1))
        assert robot._task_state.status is TaskStatus.ERROR
        assert robot._task_state.error_message == "policy step failed"
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


class TestAnAbandonedBringUpReportsHowLongItRan:
    """Releasing the arm is a terminal exit, so it settles the elapsed time too.

    ``duration`` is reset to ``0.0`` when a task starts and only a terminal
    writer settles it. The two exits that release the arm here return before the
    rollout's own loop and before the outer handler that settles every raise, so
    each has to settle its own: a bring-up that really did spend seconds on a
    motors-bus handshake and a checkpoint load otherwise reported ``0.0s``, and
    kept reporting it, since nothing writes the duration once a task is
    terminal. The stop-gate exit needs nothing of its own -
    :meth:`_honor_stop_request` settles before it releases the arm.
    """

    _SPENT = 0.15

    @pytest.mark.parametrize("exit_reached", ["cannot_be_built", "cannot_be_initialized"])
    def test_the_elapsed_time_is_reported_not_zero(self, arm, exit_reached) -> None:
        robot, spy = arm

        async def slow_boom(*args, **kwargs):
            await asyncio.sleep(self._SPENT)
            raise RuntimeError("checkpoint 'nobody/none' not found")

        async def slow_refusal(policy):
            await asyncio.sleep(self._SPENT)
            return False

        if exit_reached == "cannot_be_built":
            robot._get_policy = slow_boom
        else:
            robot._initialize_policy = slow_refusal

        asyncio.run(_execute(robot))

        assert robot._task_state.status is TaskStatus.ERROR
        assert spy.log == ["connect", "disconnect"]  # the exit under test was the one reached
        assert robot._task_state.duration >= self._SPENT
        # The figure the agent reads, now and on every later status call.
        assert "Total Duration: 0.0s" not in _text(robot.get_task_status())
