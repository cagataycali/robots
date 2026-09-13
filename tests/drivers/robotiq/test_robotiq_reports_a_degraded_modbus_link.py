"""What the 2F-85 driver says when the Modbus link stops working.

The command path is graded next door against a gripper that answers. This
module grades the other half: a link that is refused, that starts replying with
an exception, that stops replying at all, and that is closed under the driver.
None of those are hypothetical for a gripper hanging off an arm's controller -
the cable is the failure - and each has to arrive as a *reason a caller can act
on* rather than a traceback, a hang, or a success.

Three distinctions are pinned here, because collapsing any of them leaves an
operator reading the wrong fault:

* **A gripper that refused the command** (a Modbus exception reply) is not
  **a wire that failed** (a socket error). The first means the frame arrived and
  was rejected; the second means nothing arrived. The driver spells them
  differently, and a caller chasing a dead cable should not be sent to read the
  manual's exception codes.
* **A read that fails is not a robot with no joints.**
  :meth:`~strands_robots.drivers.robotiq.RobotiqDriver.get_observation` is the
  mesh's joint source for this driver
  (:func:`strands_robots.bus_access.joint_read_source`), and it is annotated
  ``-> dict[str, float]``: it has no envelope to refuse into, so it degrades to
  no joints. Raising there would take the whole peer's state publication down
  with one unplugged gripper.
* **A status report carries what the driver knows, not what it wishes.**
  ``get_status`` answers on a link that never opened, naming the failure, and it
  reports ``battery_pct`` as ``None`` rather than inventing a charge for a
  gripper powered from the arm.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from strands_robots.drivers.robotiq import RobotiqDriver
from tests.drivers.robotiq.conftest import FakeGripper

Connected = Callable[..., tuple[RobotiqDriver, FakeGripper]]
Gripper = Callable[..., FakeGripper]

#: Short enough that four stalled verbs cost under a second, long enough that a
#: loaded machine does not report a timeout the gripper did not cause.
_STALL_TIMEOUT = 0.2

#: A port nothing listens on. Privileged and unbound, so the connection is
#: refused rather than accepted by whatever else the machine happens to run.
_CLOSED_PORT = 1


def _text(envelope: dict[str, object]) -> str:
    """Return the text an error envelope carries."""
    content = envelope["content"]
    assert isinstance(content, list)
    return str(content[0]["text"])


def test_a_gripper_that_cannot_be_reached_is_named_not_raised() -> None:
    """A refused connection is a reason naming the endpoint, and no exception.

    ``connect_eagerly`` is declared ``-> str | None`` precisely so an
    unreachable gripper is a value the caller reads. The reason has to carry the
    host and port, because the one mistake it is diagnosing is a driver pointed
    at the wrong address, and a bare "connection refused" cannot tell a caller
    which address it tried.
    """
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=_CLOSED_PORT, timeout=_STALL_TIMEOUT)

    reason = driver.connect_eagerly()

    assert reason is not None, "an unreachable gripper reported success"
    assert "127.0.0.1" in reason and str(_CLOSED_PORT) in reason, f"the reason must name the endpoint: {reason!r}"
    assert not driver.is_connected
    # The failure survives the call, so a later status report can name it rather
    # than saying only that the gripper is not connected.
    status = asyncio.run(driver.get_status())
    payload = status["content"][0]["json"]
    assert payload["connected"] is False
    assert payload["connect_error"] == reason
    assert payload["battery_pct"] is None, "a gripper powered from the arm must not report a charge"
    assert "gripper" not in payload, "no reading was ever taken, so none may be reported"


def test_a_status_report_carries_the_last_reading_the_driver_took(connected: Connected) -> None:
    """``get_status`` renders the cached status, enums named rather than numbered.

    The cache is what lets an agent ask "what is the gripper doing" without
    putting another frame on the bus mid-motion, and the rendering is what makes
    the answer readable: ``object: 2`` would need the caller to own a copy of
    the manual's map.
    """
    driver, _fake = connected(object_status=2)
    driver.send_action({"gripper": 1.0})
    assert driver.read_status()["status"] == "success"

    payload = asyncio.run(driver.get_status())["content"][0]["json"]

    assert payload["connected"] is True
    assert payload["connect_error"] is None
    # The name in the payload is the name the agent and the mesh invoke this
    # driver by, so the two surfaces must not be able to disagree.
    assert payload["tool_name"] == driver.tool_name == "robotiq_2f85"
    assert payload["gripper"]["object"] == "CONTACT_CLOSING"
    assert payload["gripper"]["activation"] == "ACTIVE"
    assert payload["gripper"]["position"] == 255


@pytest.mark.parametrize(
    ("verb", "expected"),
    [
        ("send_action", "send_action: "),
        ("read_status", "read_status: "),
        ("stop_task", "stop_task: "),
    ],
)
def test_a_controller_that_starts_refusing_is_reported_by_every_verb(
    connected: Connected, verb: str, expected: str
) -> None:
    """A Modbus exception reply becomes a reason carrying the code, not a raise.

    The controller answered, so the frame reached it and was rejected: the
    exception code is the actionable part and has to survive into the envelope.
    A driver that let ``ProtocolError`` escape would put a traceback where the
    mesh command path and the agent tool dispatch both expect an envelope.
    """
    driver, fake = connected()
    fake.exception_code = 0x02  # a connected controller that starts refusing

    envelope = driver.send_action({"gripper": 1.0}) if verb == "send_action" else getattr(driver, verb)()

    assert envelope["status"] == "error", envelope
    text = _text(envelope)
    assert text.startswith(expected), f"the reason must name the verb that failed: {text!r}"
    assert "0x02" in text, f"the reason must carry what the controller said: {text!r}"


@pytest.mark.parametrize(
    ("verb", "expected"),
    [
        ("send_action", "send_action: writing to the gripper failed"),
        ("read_status", "read_status: reading the gripper failed"),
        ("stop_task", "stop_task: "),
    ],
)
def test_a_controller_that_stops_answering_is_reported_as_a_failed_wire(
    gripper: Gripper, verb: str, expected: str
) -> None:
    """A socket that times out is a wire failure, spelled as one.

    This is the failure a Modbus exception reply is not: the controller holds
    the connection open and says nothing, so no code comes back and there is
    nothing in the manual to look up. The driver must time out and say the wire
    failed - it must not block the caller waiting for a reply that is not
    coming, and it must not report the silence as the gripper refusing.
    """
    fake = gripper(starts_activated=True)
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=fake.port, timeout=_STALL_TIMEOUT)
    assert driver.connect_eagerly() is None
    fake.stall = True

    try:
        envelope = driver.send_action({"gripper": 1.0}) if verb == "send_action" else getattr(driver, verb)()

        assert envelope["status"] == "error", envelope
        text = _text(envelope)
        assert text.startswith(expected), f"a silent controller must be reported as a wire failure: {text!r}"
        assert "0x" not in text, f"there was no exception code to report: {text!r}"
    finally:
        driver.cleanup()


def test_a_controller_that_closed_the_link_names_how_much_of_the_frame_arrived(gripper: Gripper) -> None:
    """End-of-stream mid-frame is decoded, not parsed as a whole reply.

    Modbus TCP runs over a byte stream, so a reply has to be read by its
    declared length. A driver that parsed whatever one ``recv`` returned would
    decode a position out of a truncated frame - a gripper reported as open
    because the bytes ran out - and a closed link would look like a reading.
    """
    fake = gripper(starts_activated=True)
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=fake.port, timeout=2.0)
    assert driver.connect_eagerly() is None
    fake.half_close()

    try:
        envelope = driver.read_status()

        assert envelope["status"] == "error", envelope
        text = _text(envelope)
        assert "closed the connection" in text, f"a closed link must be named as one: {text!r}"
        assert "of 7 bytes" in text, f"the reason must say how much of the header arrived: {text!r}"
    finally:
        driver.cleanup()


@pytest.mark.parametrize("break_link", ["refuses", "stops_answering", "closes"])
def test_a_gripper_that_cannot_be_read_reports_no_joints_rather_than_raising(
    gripper: Gripper, break_link: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The mesh joint read degrades to no joints however the link fails.

    ``get_observation`` is annotated ``-> dict[str, float]`` and is the joint
    source the mesh publishes this peer's state from, so it has no envelope to
    refuse into. An exception here does not fail one gripper - it takes the
    state publication for every robot on that peer with it. "No joints" is the
    honest answer, and the reason belongs in the log, which is the only surface
    left that can carry it.
    """
    fake = gripper(starts_activated=True)
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=fake.port, timeout=_STALL_TIMEOUT)
    assert driver.connect_eagerly() is None
    assert driver.get_observation() == {"gripper.pos": 0.0}, "the gripper was not readable to begin with"

    if break_link == "refuses":
        fake.exception_code = 0x02
    elif break_link == "stops_answering":
        fake.stall = True
    else:
        fake.half_close()

    try:
        with caplog.at_level(logging.DEBUG, logger="strands_robots.drivers.robotiq.driver"):
            assert driver.get_observation() == {}, "an unreadable gripper must report no joints, not a stale one"
        assert [record for record in caplog.records if "joint read failed" in record.getMessage()], (
            "the reason the joint read failed was recorded nowhere"
        )
    finally:
        driver.cleanup()


def test_a_gripper_reports_no_task_in_flight_which_is_not_a_failure() -> None:
    """``get_task_status`` succeeds; a gripper having no rollout is not an error.

    The sibling verbs (``start_task``, ``run_policy``) refuse, because a caller
    asking a 1-DOF end effector to run a policy asked for something that cannot
    happen. Asking *whether* one is running is a different question with a true
    answer, so it is a success carrying ``in_flight: False`` - a task poller
    that read an error envelope here would report the gripper as broken.
    """
    envelope = RobotiqDriver().get_task_status()

    assert envelope["status"] == "success", envelope
    payload = envelope["content"][0]["json"]
    assert payload["in_flight"] is False
    assert "send_action" in payload["reason"], "the answer must name the path that does command the fingers"
