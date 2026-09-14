"""Every verb the gripper's schema declares, answered live and unconnected.

The other modules here drive :class:`~strands_robots.drivers.robotiq.RobotiqDriver`
through its Python surface - ``send_action``, ``read_status``, ``stop_task``. An
**agent** reaches none of those: it reads the ``action`` ``enum`` off
:attr:`~strands_robots.drivers.robotiq.RobotiqDriver.tool_spec` and sends one
verb to :meth:`~strands_robots.drivers.robotiq.RobotiqDriver.stream`, which is a
*router* - five declared verbs to five different methods, plus a refusal for
everything else. That router was unexecuted, so three claims it makes were
unpinned:

* **The enum is the population.** The verbs graded below are read back out of
  the schema an agent plans against, so a verb added to the enum without a
  branch fails here rather than falling into the router's ``else``.
* **``stop`` is delegated, not restated.** The branch reads
  ``envelope = self.stop_task()`` and its comment says why: "an agent that read
  a restated success here would be told the fingers stopped when the write
  failed". A gripper that answers a Modbus exception to the halt frame must
  reach the agent as an error, and that path exists only through ``stream``.
* **A verb answers the same way with no socket.** An agent invokes the tool
  before anything called ``connect_eagerly``, and the five verbs disagree on
  purpose there - two refuse the write, one refuses the read, ``status`` reports
  ``connected=False`` and ``stop`` succeeds because there is nothing to halt.

Every live case runs over a real TCP socket to :class:`FakeGripper`, which only
records a position when the frame set ``rGTO`` on an activated gripper. So the
motion assertions read what reached the wire, not what the envelope claimed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from strands.types.tools import ToolUse

from strands_robots.drivers.base import declared_verbs
from strands_robots.drivers.robotiq import STROKE_MM, RobotiqDriver
from tests.drivers.robotiq.conftest import FakeGripper

Connected = Callable[..., tuple[RobotiqDriver, FakeGripper]]

_TOOL_USE_ID = "tooluse-2f85"

#: The verbs the schema declares, in its own order. Pinned against
#: :func:`declared_verbs` by :func:`test_the_schema_declares_the_verbs_the_router_dispatches`,
#: which is what keeps the parametrized cells below from going stale or vacuous.
_DECLARED = ("open", "close", "sensors", "status", "stop")

#: Sentinel for "the agent sent no ``action`` at all", which the schema answers
#: with its ``default`` rather than a refusal.
_ABSENT = object()


def _ask(driver: RobotiqDriver, action: Any = _ABSENT) -> dict[str, Any]:
    """Invoke the driver the way the agent runtime does and return its result."""
    payload: dict[str, Any] = {} if action is _ABSENT else {"action": action}

    async def _drain() -> list[dict[str, Any]]:
        tool_use: ToolUse = {"name": "robotiq_2f85", "toolUseId": _TOOL_USE_ID, "input": payload}
        return [event async for event in driver.stream(tool_use, {})]

    events = asyncio.run(_drain())
    assert len(events) == 1, f"one invocation must yield exactly one tool result, got {len(events)}"
    assert events[0]["toolUseId"] == _TOOL_USE_ID, "the result must be attributed to the tool use that asked"
    return events[0]


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0]["text"])


def _json(envelope: dict[str, Any]) -> dict[str, Any]:
    return dict(envelope["content"][0]["json"])


def _unconnected() -> RobotiqDriver:
    """A driver that never opened a socket, as an agent's first call finds it."""
    return RobotiqDriver(tool_name="robotiq_2f85", port="127.0.0.1", tcp_port=1, timeout=0.2)


def test_the_schema_declares_the_verbs_the_router_dispatches() -> None:
    """The enum an agent plans against is the population every cell here grades."""
    spec = _unconnected().tool_spec

    assert spec["name"] == "robotiq_2f85", "an agent invokes the driver by its tool name"
    assert declared_verbs(spec) == list(_DECLARED)

    action = spec["inputSchema"]["json"]["properties"]["action"]
    assert action["default"] in _DECLARED, "the default must be a verb the router dispatches"
    # The description is the only place an agent learns what a verb *does*, so a
    # verb declared without one leaves it guessing between five names.
    for verb in _DECLARED:
        assert f"{verb}:" in action["description"], f"{verb!r} is declared but not described"


@pytest.mark.parametrize("verb", _DECLARED)
def test_every_declared_verb_is_answered_by_a_live_gripper(connected: Connected, verb: str) -> None:
    """No declared verb falls through to the refusal, and none raises."""
    driver, _fake = connected(starts_activated=True)

    envelope = _ask(driver, verb)

    assert envelope["status"] == "success", f"{verb!r} answered {envelope}"


@pytest.mark.parametrize(
    ("verb", "expected_counts", "expected_mm"),
    [
        # The direction pin at the agent surface. rPR runs backwards to
        # aperture, so a router that crossed these two would answer a tidy
        # success while closing a gripper the agent asked to let go.
        ("open", 0, STROKE_MM),
        ("close", 255, 0.0),
    ],
)
def test_open_and_close_reach_the_wire_as_full_aperture_and_full_grasp(
    connected: Connected, verb: str, expected_counts: int, expected_mm: float
) -> None:
    """The gripper records the count, so the fingers really went where asked."""
    driver, fake = connected(starts_activated=True)

    envelope = _ask(driver, verb)

    assert _json(envelope)["aperture_mm"] == expected_mm
    assert fake.commanded == [expected_counts], f"{verb!r} did not reach the gripper as {expected_counts}"


def test_stop_halts_the_fingers_where_they_are(connected: Connected) -> None:
    """``stop`` clears ``rGTO`` and commands no new position of its own."""
    driver, fake = connected(starts_activated=True)
    assert _ask(driver, "close")["status"] == "success"

    envelope = _ask(driver, "stop")

    assert envelope["status"] == "success"
    assert "robotiq_2f85" in _text(envelope)
    assert fake.writes[-1]["activate"] == 1, "a halt must not deactivate the gripper"
    assert fake.writes[-1]["go_to"] == 0, "the halt frame must clear rGTO"
    assert fake.commanded == [255], "the halt moved the fingers instead of stopping them"


def test_stop_reports_a_gripper_that_refused_the_halt(connected: Connected) -> None:
    """A controller that answers an exception to the halt reaches the agent as one.

    The verdict is ``stop_task``'s, forwarded verbatim: an envelope written
    beside the write could only restate the intent, and the agent would read
    ``success`` for fingers that never stopped.
    """
    driver, fake = connected(starts_activated=True)
    fake.exception_code = 0x04

    envelope = _ask(driver, "stop")

    assert envelope["status"] == "error", f"a refused halt reported {envelope}"
    assert "stop_task" in _text(envelope), "the refusal must come from the surface that establishes it"
    assert "0x04" in _text(envelope), f"the Modbus exception is what a caller acts on: {_text(envelope)}"


@pytest.mark.parametrize("verb", ["home", "sensor", "STOP", "", None])
def test_an_undeclared_verb_is_refused_and_nothing_reaches_the_gripper(connected: Connected, verb: Any) -> None:
    """A typo, a sibling's spelling or a non-string is named, not dispatched."""
    driver, fake = connected(starts_activated=True)

    envelope = _ask(driver, verb)

    assert envelope["status"] == "error"
    assert repr(verb) in _text(envelope), f"the refusal must quote what arrived: {_text(envelope)}"
    assert str(list(_DECLARED)) in _text(envelope), "the refusal must name the verbs that do work"
    assert fake.writes == [], f"an undeclared verb wrote to the gripper: {fake.writes}"


def test_an_absent_action_reads_the_sensors_the_schema_defaults_to(connected: Connected) -> None:
    """A tool use with no ``action`` is answered by the schema's default verb."""
    driver, fake = connected(starts_activated=True)
    default = driver.tool_spec["inputSchema"]["json"]["properties"]["action"]["default"]

    absent = _ask(driver)

    assert absent["status"] == "success"
    assert "aperture_mm" in _json(absent), f"{default!r} must answer a sensor reading, not connection state"
    assert _json(absent) == _json(_ask(driver, default)), f"an absent action did not run {default!r}"
    assert fake.commanded == [], "reading the sensors must not move the fingers"


@pytest.mark.parametrize(
    ("verb", "expected_status", "expected_text"),
    [
        # The five verbs disagree here on purpose: a write the gripper never saw
        # cannot be reported as done, a read has no value to return, connection
        # state is exactly what an unconnected driver still knows, and a halt
        # with nothing to halt has already succeeded.
        ("open", "error", "send_action: not connected"),
        ("close", "error", "send_action: not connected"),
        ("sensors", "error", "read_status: not connected"),
        ("status", "success", None),
        ("stop", "success", "nothing to stop"),
    ],
)
def test_every_declared_verb_answers_without_a_connection(
    verb: str, expected_status: str, expected_text: str | None
) -> None:
    """An agent's first call arrives before ``connect_eagerly``; none of it raises."""
    driver = _unconnected()

    envelope = _ask(driver, verb)

    assert envelope["status"] == expected_status, f"{verb!r} answered {envelope}"
    if expected_text is None:
        assert _json(envelope)["connected"] is False, "status must report the socket it does not have"
    else:
        assert expected_text in _text(envelope), f"{verb!r} answered {_text(envelope)!r}"
