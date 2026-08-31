"""Stopping a flying robot means landing it, and the SDK ordering is load-bearing.

Two ``cflib`` calls sound alike and do opposite things. ``send_stop_setpoint``
sets every motor to zero: an airborne Crazyflie drops out of the air.
``HighLevelCommander.land`` descends under control. The
:class:`~strands_robots.drivers.base.HardwareDriver` contract's ``stop`` is
documented as "stop motion, leaving the robot connected", and on an airframe that
cannot hold still without a setpoint stream the only motion-free state is on the
ground - so ``stop`` must land.

The second pin is an ordering the SDK requires and a reader would not guess.
While the low-level commander is streaming it owns the setpoint priority, and a
``land`` issued underneath it is ignored: the aircraft keeps flying the last
twist. ``Commander.send_notify_setpoint_stop`` is what hands priority back, so it
must precede the ``land``. That is a relation between calls on two different SDK
objects, which is why the fake records them on one ordered list.
"""

from __future__ import annotations

import asyncio

import pytest


class TestStopLands:
    """The verb the contract calls ``stop``, on an aircraft."""

    def test_stop_never_reaches_the_motor_cut(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        driver, _, reason = connected(setpoint_hz=100)
        assert reason is None
        driver.send_action({"vx": 0.2, "z": 0.5})
        asyncio.run(driver.stop())

        assert recorder.count("commander.send_stop_setpoint") == 0, (
            "stop() must not cut the motors; an airborne aircraft would fall. That is "
            "emergency_stop(), which a caller has to name."
        )
        assert recorder.count("high_level.land") == 1, "stop() must bring the aircraft down"

    def test_stop_task_lands_too(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """Stopping the stream without landing is the fall this driver avoids."""
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        envelope = driver.stop_task()

        assert envelope["status"] == "success"
        assert envelope["content"][0]["json"]["stopped"] == "setpoint_stream"
        assert recorder.count("high_level.land") == 1
        assert recorder.count("commander.send_stop_setpoint") == 0

    def test_cleanup_lands_before_it_closes_the_link(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """A descent needs the link, so releasing it first would abandon the aircraft."""
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"z": 0.5})
        driver.cleanup()

        names = recorder.names()
        assert "high_level.land" in names and "close_link" in names
        assert names.index("high_level.land") < names.index("close_link")
        assert recorder.count("commander.send_stop_setpoint") == 0


class TestThePriorityHandoverPrecedesTheLand:
    """``send_notify_setpoint_stop`` first, or the ``land`` is ignored."""

    def test_the_handover_is_sent_and_it_is_sent_first(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        assert driver.land()["status"] == "success"

        names = recorder.names()
        assert "commander.send_notify_setpoint_stop" in names, (
            "without this the low-level stream keeps the setpoint priority and the land is ignored"
        )
        assert names.index("commander.send_notify_setpoint_stop") < names.index("high_level.land")

    def test_the_repeater_is_stopped_before_the_handover(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """Otherwise the background thread re-latches a twist mid-descent."""
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        driver.land()

        names = recorder.names()
        handover = names.index("commander.send_notify_setpoint_stop")
        assert "commander.send_hover_setpoint" not in names[handover:], (
            f"a setpoint was sent after the priority handover: {names[handover:]}"
        )
        assert driver.get_task_status()["content"][0]["json"]["streaming"] is False


class TestCuttingTheMotorsIsItsOwnVerb:
    """Reachable only by name, and it says what it does."""

    def test_emergency_stop_cuts_the_motors(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"z": 0.5})
        envelope = driver.emergency_stop()

        assert envelope["status"] == "success"
        assert recorder.count("commander.send_stop_setpoint") == 1
        assert recorder.count("high_level.land") == 0
        assert "falls" in envelope["content"][0]["json"]["note"], (
            "the envelope must say what cutting the motors does to an airborne aircraft"
        )

    def test_the_agent_tool_schema_cannot_reach_it(self, connected) -> None:  # type: ignore[no-untyped-def]
        """An agent picking a verb off the schema must not be able to pick this one."""
        driver, _, _ = connected()
        verbs = driver.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert "land" in verbs, "the agent needs a way to bring the aircraft down"
        assert not {"stop", "emergency_stop", "kill"} & set(verbs), f"the schema offers a motor-cut verb: {verbs}"


class TestTheAgentVerbReportsTheDescentVerdict:
    """The agent gets the halt outcome, not an acknowledgement of the request.

    ``stream``'s stop branch returns ``stop_task()``'s envelope verbatim. An
    envelope built beside the call could only restate the intent ("asked it to
    land") while the descent itself can be refused, and an agent that reads
    success for a refused descent believes the aircraft is coming down.
    """

    def test_a_successful_descent_is_reported_as_one(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        result = asyncio.run(_invoke(driver, "land"))

        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"] == "land"
        assert recorder.count("high_level.land") == 1

    def test_a_refused_descent_is_not_reported_as_success(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The link is gone, so there is no descent to acknowledge."""
        driver, _, _ = connected()
        driver.cleanup()
        recorder.calls.clear()
        result = asyncio.run(_invoke(driver, "land"))

        assert result["status"] == "error", "an agent that reads success here believes the aircraft is coming down"
        assert "stop_task" in result["content"][0]["text"]
        assert recorder.count("high_level.land") == 0


async def _invoke(driver, action: str) -> dict:  # type: ignore[no-untyped-def]
    """Run one agent tool call and return its single result."""
    async for event in driver.stream({"toolUseId": "t1", "input": {"action": action}}, {}):
        return event
    raise AssertionError("stream yielded nothing")


class TestARefusedDurationSendsNothing:
    """The refusal precedes the descent, so a bad duration is not a half-land."""

    @pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf"), "2.0", None])
    def test_an_unusable_duration_is_refused_before_any_call(self, connected, recorder, duration) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected()
        envelope = driver.land(duration=duration)

        assert envelope["status"] == "error"
        assert "duration" in envelope["content"][0]["text"]
        assert recorder.count("high_level.land") == 0
