"""A controller that raises on the wire is reported, never propagated.

Every UR verb reaches the arm through the two ``ur_rtde`` interfaces, and a
compiled SDK talking to a controller over a network link raises rather than
returning a value when that link drops mid-register-read: ``OSError`` from the
socket, ``RuntimeError`` from ``ur_rtde``'s own error path. Each such call site
in :mod:`strands_robots.drivers.ur` is wrapped, and this module pins that the
wrapping answers in *that verb's own vocabulary* rather than in one shape for
all of them:

* a verb that owes the agent an envelope returns a refusal naming the operation
  that raised, so the failure is readable in a transcript instead of arriving as
  a traceback past the tool dispatcher;
* a verb declared ``-> None`` (``stop``, ``cleanup``) absorbs the raise and
  still completes its teardown, because a link that is already gone is the very
  condition those verbs exist to clean up after;
* ``get_status`` stays a *success* whose mode fields read ``None``, since
  reachability is exactly what a caller asks it for when the link is sick;
* ``connect_eagerly`` is declared ``-> str | None`` and returns the reason, so a
  controller that cannot report its mode is a connection that did not happen.

The doubles are installed as the real ``rtde_control`` / ``rtde_receive``
importables (see ``conftest.py``), so these cells drive the driver's own SDK
resolution and its own error handling, not a patch over either.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.drivers.ur import JOINT_NAMES, URDriver
from tests.mocks.ur_rtde import MEASURED_Q, FakeReceive, FakeRTDE, json_of, text_of

HOST = "192.168.1.10"


def _boom(*_args: object, **_kwargs: object) -> Any:
    """Fail the way a dropped RTDE link fails: raise, never return."""
    raise OSError("RTDE link went away mid-register-read")


def _connected(fake: FakeRTDE) -> URDriver:
    """A driver connected over the doubles, ready for one broken call."""
    driver = URDriver(tool_name="ur5e", port=HOST)
    assert driver.connect_eagerly() is None
    return driver


def _reachable_setpoint() -> dict[str, float]:
    """A setpoint one step off the measured pose, so the step gate admits it."""
    return {name: q + 0.001 for name, q in zip(JOINT_NAMES, MEASURED_Q, strict=True)}


#: ``(label, interface, method, drive, fragment)`` - the verbs that owe the
#: agent an envelope. ``interface``/``method`` name the SDK call made to raise;
#: ``fragment`` is the operation the refusal must name, so a reader learns which
#: register read or motion command failed rather than only that something did.
REFUSING_SITES: tuple[tuple[str, str, str, Any, str], ...] = (
    (
        "send_action names the motion command that raised",
        "control",
        "servoJ",
        lambda d: d.send_action(_reachable_setpoint()),
        "send_action: servoJ failed:",
    ),
    (
        "state names a failed velocity/pose read",
        "receive",
        "getActualQd",
        lambda d: d.state(),
        "state: the controller's RTDE read failed:",
    ),
    (
        "state names a failed joint read separately",
        "receive",
        "getActualQ",
        lambda d: d.state(),
        "state: the controller's joint read failed:",
    ),
    (
        "stop_task names the deceleration command that raised",
        "control",
        "servoStop",
        lambda d: d.stop_task(),
        "stop_task: the controller refused servoStop:",
    ),
)


@pytest.mark.parametrize(
    ("interface", "method", "drive", "fragment"),
    [row[1:] for row in REFUSING_SITES],
    ids=[row[0] for row in REFUSING_SITES],
)
def test_a_verb_that_owes_an_envelope_refuses_the_raise(
    fake_rtde: FakeRTDE,
    monkeypatch: pytest.MonkeyPatch,
    interface: str,
    method: str,
    drive: Any,
    fragment: str,
) -> None:
    """The raise becomes a refusal naming the SDK call, not a traceback.

    A driver verb is reached through the agent's tool dispatcher, which owes the
    model an envelope; an exception crossing that boundary is reported as a tool
    crash with no statement of which register read or motion command failed.
    """
    driver = _connected(fake_rtde)
    monkeypatch.setattr(getattr(fake_rtde, interface), method, _boom)

    result = drive(driver)

    assert result["status"] == "error"
    assert fragment in text_of(result)


def test_get_status_still_reports_reachability_when_the_mode_read_raises(
    fake_rtde: FakeRTDE, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sick link leaves ``get_status`` a success with unknown modes.

    ``get_status`` is what a caller reaches for *because* the arm is
    misbehaving, so refusing it would withhold the reachability answer at
    exactly the moment it is wanted. The two mode fields read ``None`` -
    unknown, stated as unknown - rather than a stale or invented mode name.
    """
    driver = _connected(fake_rtde)
    monkeypatch.setattr(fake_rtde.receive, "getRobotMode", _boom)

    status = asyncio.run(driver.get_status())

    assert status["status"] == "success"
    body = json_of(status)
    assert body["connected"] is True
    assert body["robot_mode"] is None
    assert body["safety_mode"] is None


def test_stop_absorbs_a_refused_servo_stop(fake_rtde: FakeRTDE, monkeypatch: pytest.MonkeyPatch) -> None:
    """``stop`` is declared ``-> None`` and stays silent on a raise.

    The e-stop rail calls this on every registered motion source; one arm whose
    link has already dropped must not abort the sweep before the others are
    told to halt.
    """
    driver = _connected(fake_rtde)
    monkeypatch.setattr(fake_rtde.control, "servoStop", _boom)

    asyncio.run(driver.stop())  # the claim is that this returns rather than raises

    # A halt that could not be delivered is not a disconnection: the interfaces
    # stay adopted so a caller can retry or read the arm's mode.
    assert driver.is_connected is True


def test_stop_on_an_unconnected_driver_reaches_no_interface(fake_rtde: FakeRTDE) -> None:
    """With no control interface there is nothing to decelerate, and no raise."""
    driver = URDriver(tool_name="ur5e", port=HOST)

    asyncio.run(driver.stop())  # returns rather than raising

    assert not fake_rtde.controls


def test_cleanup_releases_both_interfaces_though_every_call_raises(
    fake_rtde: FakeRTDE, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown completes when the deceleration *and* both disconnects raise.

    ``cleanup`` runs on a path that is already recovering, so a controller that
    cannot be spoken to must not strand the interfaces: the driver drops its
    references either way, and a second cleanup stays a no-op.
    """
    driver = _connected(fake_rtde)
    control, receive = fake_rtde.control, fake_rtde.receive
    monkeypatch.setattr(control, "servoStop", _boom)
    monkeypatch.setattr(control, "disconnect", _boom)
    monkeypatch.setattr(receive, "disconnect", _boom)

    driver.cleanup()  # the claim is that this completes rather than raising

    assert driver.is_connected is False
    # Idempotent: the references were dropped, so nothing is spoken to twice.
    driver.cleanup()


def test_connect_reports_a_controller_that_cannot_report_its_mode(
    fake_rtde: FakeRTDE, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mode read that raises is a connection that did not happen.

    ``connect_eagerly`` is declared ``-> str | None``, and the mode
    interrogation is the gate that exists because a UR controller in a
    protective stop accepts an RTDE control connection and then moves nothing.
    A raise there is no more a usable arm than a stopped mode is, so it takes
    the same route: the reason is returned, and the driver stays disconnected.
    """
    monkeypatch.setattr(FakeReceive, "getRobotMode", _boom)
    driver = URDriver(tool_name="ur5e", port=HOST)

    reason = driver.connect_eagerly()

    assert reason is not None
    assert "did not report its mode" in reason
    assert driver.is_connected is False
