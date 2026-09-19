"""A bridge's read verb refuses a wait it cannot honor, and reads nothing.

``get_pose`` and ``get_scan`` take one knob, ``timeout``, and hand it to a
transport as a wait budget. Only a positive finite value is a wait: on a
transport that is already connected, ``0``/``-1``/``nan`` returns the moment it
is asked, so an ungraded value is reported as a *successful* read of zero
samples - the caller is told the topic was silent when in fact the wait was
never made. A control loop that computes a remaining deadline and underflows is
exactly how such a value arrives, and it used to be refused by name.

The domain therefore belongs to the bridge, next to the ``drive`` knobs it
already grades, not to the transport: a transport reached by both an agent tool
and a library class honors the wait it is handed and states no domain of its
own. The rows below are every read a bridge can actually make - ``RtpsRobot``
configures no odometry or scan topic, so it refuses on the topic first.

Each case replaces the transport with a recorder, so what is graded is the
*bridge's* refusal rather than one that happens to live downstream: a bridge
whose only guard is the tool it used to forward through refuses nothing here,
which is what a second caller of the same transport inherits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import strands_robots.mesh.ackermann_robot as ackermann_mod
import strands_robots.mesh.ros_bridge as ros_bridge_mod
import strands_robots.mesh.rosbridge_robot as rosbridge_mod
from strands_robots.utils import positive_finite_number_error

#: Values no wait budget can express. ``inf`` matters as much as ``0``: it passes
#: a bare ``timeout > 0`` test and then waits forever on a topic that is silent.
UNUSABLE_TIMEOUTS: list[Any] = [0, 0.0, -1.0, float("nan"), float("inf"), "2", None, True, [5.0]]

#: (label, module, forwarded transport symbol, bridge factory, read verb name).
_READS: list[tuple[str, Any, str, Callable[[], Any], str]] = [
    (
        "ros_bridge",
        ros_bridge_mod,
        "ros_action",
        lambda: ros_bridge_mod.RosBridgedRobot("rover", "/cmd_vel", "/odom", scan_topic="/scan"),
        "get_pose",
    ),
    (
        "ros_bridge",
        ros_bridge_mod,
        "ros_action",
        lambda: ros_bridge_mod.RosBridgedRobot("rover", "/cmd_vel", "/odom", scan_topic="/scan"),
        "get_scan",
    ),
    (
        "rosbridge",
        rosbridge_mod,
        "rosbridge_action",
        lambda: rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", scan_topic="/scan"),
        "get_pose",
    ),
    (
        "rosbridge",
        rosbridge_mod,
        "rosbridge_action",
        lambda: rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", scan_topic="/scan"),
        "get_scan",
    ),
    (
        "ackermann",
        ackermann_mod,
        "ros_action",
        lambda: ackermann_mod.AckermannRosRobot("car", "/servo", scan_topic="/scan"),
        "get_scan",
    ),
]


class _Recorder:
    """Records each forwarded transport call, so a read that happened is visible."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "{}"}]}


def _read(monkeypatch: pytest.MonkeyPatch, row: tuple[Any, ...], timeout: Any) -> tuple[dict[str, Any], _Recorder]:
    _label, module, symbol, factory, verb = row
    rec = _Recorder()
    monkeypatch.setattr(module, symbol, rec)
    return getattr(factory(), verb)(timeout=timeout), rec


@pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS, ids=repr)
@pytest.mark.parametrize("row", _READS, ids=lambda row: f"{row[0]}.{row[4]}")
def test_a_read_refuses_an_unusable_timeout_and_reaches_no_transport(
    monkeypatch: pytest.MonkeyPatch, row: tuple[Any, ...], value: Any
) -> None:
    """The refusal names the verb the caller invoked, and nothing is read.

    Asserting the shared helper's exact text is what makes this a parity check
    across the three bridges: one that happened to fail for another reason - a
    raw ``OverflowError`` out of a wait, or an empty success - still fails here.
    """
    expected = positive_finite_number_error(value, "timeout", row[4])
    assert expected is not None, "probe value must be outside the domain"

    result, rec = _read(monkeypatch, row, value)

    assert result["status"] == "error", f"{row[0]}.{row[4]} accepted timeout={value!r}"
    assert str(result["content"][0]["text"]) == expected
    assert rec.calls == [], f"{row[0]}.{row[4]} reached the transport for timeout={value!r}"


@pytest.mark.parametrize("row", _READS, ids=lambda row: f"{row[0]}.{row[4]}")
def test_a_usable_timeout_is_forwarded_unchanged(monkeypatch: pytest.MonkeyPatch, row: tuple[Any, ...]) -> None:
    """The premise: the guard refuses only what cannot be honored."""
    result, rec = _read(monkeypatch, row, 2.5)

    assert result["status"] == "success"
    assert [call["timeout"] for call in rec.calls] == [2.5]
