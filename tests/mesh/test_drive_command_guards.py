"""A velocity command a bridge cannot honor is refused before it reaches the wire.

:meth:`RosBridgedRobot.drive` and :meth:`RtpsRobot.drive` are the calls that
physically move a mobile base. Both derive the published message count from
``round(duration * publish_rate)`` and forward the velocity components verbatim,
so a value that cannot be honored has one of two silent outcomes: a plausible
burst is published anyway (``duration=0`` becomes a single full-speed command
via the ``max(1, ...)`` floor), or the arithmetic raises a bare
``ValueError``/``OverflowError``/``TypeError`` out of a method whose contract is
a ``{"status": ...}`` result dict - and both ``drive`` methods are exposed to an
agent as a tool, where raising escapes the dispatch contract entirely.

``use_ros``/``use_rtps`` cannot cover this: they validate the topic and interface
type, but ``duration`` never reaches them (they receive only the derived count)
and neither validates ``count`` or the Twist field values. The guards therefore
belong on the bridge, and these tests assert the two transports agree on exactly
which commands are refusable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

import strands_robots.mesh.ros_bridge as ros_mod
import strands_robots.mesh.rtps_robot as rtps_mod
from strands_robots.mesh import RosBridgedRobot, RtpsRobot


class _Wire:
    """Stands in for the transport, recording every published message batch."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


# (label, module, forwarded-symbol name, factory) for each transport.
_TRANSPORTS: list[tuple[str, Any, str, Callable[..., Any]]] = [
    ("ros", ros_mod, "use_ros", lambda **kw: RosBridgedRobot("tb", "/cmd_vel", "/odom", **kw)),
    ("rtps", rtps_mod, "use_rtps", lambda **kw: RtpsRobot("tb", "/cmd_vel", **kw)),
]
_TRANSPORT_IDS = [t[0] for t in _TRANSPORTS]

# Values no message count can express: a zero/negative hold, a non-finite hold,
# and a hold that is not a number at all.
_BAD_DURATIONS = [0, 0.0, -5, -0.25, float("nan"), float("inf"), float("-inf"), True, "2", [1.0]]
# ``count`` is the horizon only when no duration is given.
_BAD_COUNTS = [0, -1, 2.7, float("nan"), float("inf"), True, "3", None]
# A velocity is signed, so only non-finite and non-numeric values are refusable.
_BAD_VELOCITIES = [float("nan"), float("inf"), float("-inf"), "1.0", None, [1.0], True]


@pytest.fixture(params=_TRANSPORTS, ids=_TRANSPORT_IDS)
def bridge(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Wire]:
    """A bridge for one transport plus the wire recorder it forwards to."""
    _label, module, symbol, factory = request.param
    wire = _Wire()
    monkeypatch.setattr(module, symbol, wire)
    return factory(), wire


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


class TestRefusedCommandNeverReachesTheWire:
    """A refusable value yields an error result and publishes nothing."""

    @pytest.mark.parametrize("duration", _BAD_DURATIONS)
    def test_a_hold_no_message_count_expresses_is_refused(self, bridge: tuple[Any, _Wire], duration: Any) -> None:
        robot, wire = bridge
        result = robot.drive(linear=0.5, duration=duration)
        assert result["status"] == "error"
        assert "duration" in _text(result)
        assert wire.calls == []

    @pytest.mark.parametrize("count", _BAD_COUNTS)
    def test_a_message_count_that_publishes_nothing_is_refused(self, bridge: tuple[Any, _Wire], count: Any) -> None:
        robot, wire = bridge
        result = robot.drive(linear=0.5, count=count)
        assert result["status"] == "error"
        assert "count" in _text(result)
        assert wire.calls == []

    @pytest.mark.parametrize("velocity", _BAD_VELOCITIES)
    @pytest.mark.parametrize("param", ["linear", "angular"])
    def test_a_velocity_the_controller_cannot_integrate_is_refused(
        self, bridge: tuple[Any, _Wire], param: str, velocity: Any
    ) -> None:
        robot, wire = bridge
        result = robot.drive(**{param: velocity})
        assert result["status"] == "error"
        assert param in _text(result)
        assert wire.calls == []

    def test_the_first_refusable_parameter_is_named(self, bridge: tuple[Any, _Wire]) -> None:
        """Several bad values at once name one parameter, not a merged message."""
        robot, wire = bridge
        result = robot.drive(linear=float("nan"), duration=-1)
        assert result["status"] == "error"
        assert "linear" in _text(result) and "duration" not in _text(result)
        assert wire.calls == []


class TestHonoredCommandsStillPublish:
    """The guards do not narrow the set of commands that already worked."""

    def test_a_duration_sizes_the_burst_from_the_publish_rate(self, bridge: tuple[Any, _Wire]) -> None:
        robot, wire = bridge
        assert robot.drive(linear=1.0, duration=1.5)["status"] == "success"
        assert wire.calls[0]["count"] == 15

    def test_a_hold_shorter_than_one_period_still_sends_the_command_once(self, bridge: tuple[Any, _Wire]) -> None:
        """The ``max(1, ...)`` floor is a rounding rule for a valid duration."""
        robot, wire = bridge
        assert robot.drive(linear=1.0, duration=0.01)["status"] == "success"
        assert wire.calls[0]["count"] == 1

    @pytest.mark.parametrize("linear,angular", [(-1.5, 0.0), (0.0, -2.0), (0.0, 0.0), (1e-9, 1e9)])
    def test_a_signed_velocity_is_published_verbatim(
        self, bridge: tuple[Any, _Wire], linear: float, angular: float
    ) -> None:
        robot, wire = bridge
        assert robot.drive(linear=linear, angular=angular)["status"] == "success"
        assert wire.calls[0]["fields"] == {"linear": {"x": linear}, "angular": {"z": angular}}

    def test_a_count_the_call_never_reads_is_not_refused(self, bridge: tuple[Any, _Wire]) -> None:
        """``duration`` supersedes ``count``, so an unread ``count`` is not the horizon."""
        robot, wire = bridge
        assert robot.drive(linear=1.0, duration=2.0, count=0)["status"] == "success"
        assert wire.calls[0]["count"] == 20

    def test_stop_still_publishes_zero_velocity(self, bridge: tuple[Any, _Wire]) -> None:
        robot, wire = bridge
        assert robot.stop()["status"] == "success"
        assert wire.calls[0]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


class TestPublishRateRefusedAtConstruction:
    """A rate that cannot pace the burst is refused where it is supplied."""

    @pytest.mark.parametrize("publish_rate", [0, -10, float("nan"), float("inf"), True, "10", None])
    @pytest.mark.parametrize("factory", [t[3] for t in _TRANSPORTS], ids=_TRANSPORT_IDS)
    def test_a_rate_that_cannot_pace_the_burst_is_refused(self, factory: Any, publish_rate: Any) -> None:
        with pytest.raises(ValueError, match="publish_rate"):
            factory(publish_rate=publish_rate)

    @pytest.mark.parametrize("factory", [t[3] for t in _TRANSPORTS], ids=_TRANSPORT_IDS)
    def test_a_fractional_rate_is_usable_and_normalized(self, factory: Any) -> None:
        """A rate is continuous - 2.5 Hz is a real setting, unlike a frame count."""
        robot = factory(publish_rate=2.5)
        assert isinstance(robot.publish_rate, float)
        assert robot.publish_rate == pytest.approx(2.5)


class TestAgentToolContract:
    """The bound ``drive_*`` agent tool returns a result; it never raises."""

    @pytest.mark.parametrize("kwargs", [{"duration": float("nan")}, {"duration": 0}, {"linear": float("inf")}])
    def test_the_bound_drive_tool_reports_instead_of_raising(
        self, bridge: tuple[Any, _Wire], kwargs: dict[str, Any]
    ) -> None:
        robot, wire = bridge
        drive_tool: Any = next(t for t in robot.tools if t.tool_name.startswith("drive_"))
        result = drive_tool(**kwargs)
        assert result["status"] == "error"
        assert wire.calls == []


class TestTransportsAgreeOnTheAcceptedDomain:
    """One rule per parameter, so the two transports cannot drift apart."""

    @pytest.mark.parametrize("value", [*_BAD_DURATIONS, 1.5, 0.01, 60])
    def test_both_transports_return_the_same_duration_verdict(
        self, monkeypatch: pytest.MonkeyPatch, value: Any
    ) -> None:
        verdicts = {}
        for label, module, symbol, factory in _TRANSPORTS:
            wire = _Wire()
            monkeypatch.setattr(module, symbol, wire)
            verdicts[label] = factory().drive(linear=0.5, duration=value)["status"]
        assert verdicts["ros"] == verdicts["rtps"], f"verdicts differ for duration={value!r}: {verdicts}"

    @pytest.mark.parametrize("value", [*_BAD_VELOCITIES, 0.0, -1.0, 2.5])
    def test_both_transports_return_the_same_velocity_verdict(
        self, monkeypatch: pytest.MonkeyPatch, value: Any
    ) -> None:
        verdicts = {}
        for label, module, symbol, factory in _TRANSPORTS:
            wire = _Wire()
            monkeypatch.setattr(module, symbol, wire)
            verdicts[label] = factory().drive(linear=value)["status"]
        assert verdicts["ros"] == verdicts["rtps"], f"verdicts differ for linear={value!r}: {verdicts}"


class TestFiniteNumberErrorDomain:
    """Unit contract of the shared signed-scalar guard."""

    @pytest.mark.parametrize("value", [0, 0.0, -1, 1e300, -1e300, 2.5])
    def test_any_finite_real_of_either_sign_is_accepted(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        assert finite_number_error(value, "linear", "drive") is None

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False, "1.0", None, [1.0]])
    def test_a_non_finite_or_non_real_value_is_refused_by_name(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        message = finite_number_error(value, "linear", "drive")
        assert message is not None
        assert message.startswith("drive: linear must be a finite number")

    def test_a_numpy_scalar_velocity_is_accepted(self) -> None:
        """A policy action element arrives as a NumPy scalar, not a Python float."""
        import numpy as np

        from strands_robots.utils import finite_number_error

        assert finite_number_error(np.float32(-0.25), "linear", "drive") is None
        assert finite_number_error(np.float64("nan"), "linear", "drive") is not None
