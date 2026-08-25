"""An :class:`RtpsRobot` command must reach the ``use_rtps`` operator gate.

The sibling of ``tests/mesh/test_ros_bridge_command_gate.py``, for the other
transport that reaches a ROS 2 graph. ``use_rtps`` gained the shared
operator-approval gate of :mod:`strands_robots.tools._command_gate`, so an
:class:`RtpsRobot` whose transport dropped the injected context would turn its
whole command surface - including the ``stop`` halt and the trailing zero of a
timed drive - into a per-call refusal whose only offered remedy is the blanket
``BYPASS_TOOL_CONSENT``.

``tests/mesh/test_rtps_robot.py`` cannot see that: it patches the ``use_rtps``
symbol at the mesh boundary, which is the boundary the gate lives behind. These
tests keep the real ``use_rtps`` and double the DDS backend beneath it instead -
the same boundary ``tests/tools/test_use_rtps.py`` doubles - so the gate and the
transport wiring under test both run unmodified while no sample reaches a real
DDS graph.

What that buys over the structural checks in ``tests/mesh/test_mobile_base.py``:
those decide forwarding by scanning the transport's source for a
``tool_context=`` keyword, so forwarding a literal ``None`` reads as forwarding
the parameter and the reported defect can be reintroduced by a one-token edit
with nothing failing. The cases here assert an outcome, which makes the
operator's answer an input to what the robot does.

The doubles resolve the interface the robot declares and refuse the rest, the
way :func:`strands_robots.rtps.idl.get_type` does. A resolver that answered every
type string would hand back a velocity message whatever the robot asked for, so
``cmd_vel_type`` would stop being an input to any assertion here and a robot that
declared an interface with nowhere to put ``linear.x`` would still read as having
driven.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.rtps.idl as idl_mod
import strands_robots.tools.use_rtps as rtps_mod
from strands_robots.mesh import RtpsRobot

_TWIST = "geometry_msgs/msg/Twist"
# In the bundle, and the wrong shape for a velocity: a robot declaring it
# resolves an interface and then has nowhere to put ``linear.x``.
_POINT = "geometry_msgs/msg/Point"
# Well-formed enough for the tool's ``pkg/msg/Name`` check, and not a type the
# bundle carries - so the resolver is what has to refuse it.
_OUTSIDE_THE_BUNDLE = "geometry_msgs/msg/Wrench"


# Module-level fake IDL dataclasses so ``typing.get_type_hints`` can resolve the
# nested field types against module globals, mirroring the real bundle.
@dataclasses.dataclass
class _Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclasses.dataclass
class _Twist:
    linear: _Vec3 = dataclasses.field(default_factory=_Vec3)
    angular: _Vec3 = dataclasses.field(default_factory=_Vec3)


@dataclasses.dataclass
class _Point:
    """``geometry_msgs/msg/Point``: three floats, and no velocity field at all.

    The same shape as :class:`_Vec3` under its own name, so a refusal reports the
    interface the robot declared rather than one of its members.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


# What the doubled resolver carries. Keyed by the same type strings the real
# bundle uses, so a robot's ``cmd_vel_type`` decides which dataclass a sample is
# built from - exactly as it does against ``strands_robots.rtps.idl.REGISTRY``.
_BUNDLE: dict[str, type] = {_TWIST: _Twist, _POINT: _Point}


class _FakeWriter:
    """A DDS writer double that also records the surface it was opened for.

    ``use_rtps`` resolves the declared interface and then asks the backend for a
    writer on ``(topic, type)``, so recording that pair is what lets a case
    assert the robot's declared ``cmd_vel_type`` reached the wire and not only
    its topic.
    """

    def __init__(self) -> None:
        self.written: list[Any] = []
        self.requested: list[tuple[str, str]] = []

    def write(self, sample: Any) -> None:
        self.written.append(sample)

    def open(self, ros_topic: str, ros_type: str) -> _FakeWriter:
        """Stand in for ``_backend.writer``: record the surface, hand back self."""
        self.requested.append((ros_topic, ros_type))
        return self


class _FakeBundle:
    """The IDL bundle double: resolves :data:`_BUNDLE`, refuses everything else.

    :func:`strands_robots.rtps.idl.get_type` raises ``KeyError`` for a type it
    does not carry, and ``use_rtps`` turns that into a reported error. A double
    that resolved every string would be strictly more permissive than the
    resolver it stands in for.
    """

    def __init__(self) -> None:
        self.resolved: list[str] = []

    def get_type(self, ros_type: str) -> type:
        self.resolved.append(ros_type)
        if ros_type not in _BUNDLE:
            raise KeyError(f"{ros_type!r} is not in the RTPS IDL bundle. Known types: {', '.join(sorted(_BUNDLE))}.")
        return _BUNDLE[ros_type]


def _install_doubles(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeWriter, _FakeBundle]:
    """Stand the DDS backend and the IDL bundle down for one test.

    Both gate env vars short-circuit the operator prompt, so an ambient
    ``BYPASS_TOOL_CONSENT`` (common in agent/automation shells) would make these
    assertions pass without the gate ever running. Cases that need them opt in
    explicitly.
    """
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
    writer, bundle = _FakeWriter(), _FakeBundle()
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rtps_mod._backend, "writer", writer.open)
    # publish sleeps for the reader settle and for the inter-message period.
    monkeypatch.setattr(rtps_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(idl_mod, "get_type", bundle.get_type)
    return writer, bundle


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _turtle() -> RtpsRobot:
    """A robot whose ``cmd_vel`` is a blocklisted surface.

    ``/turtle1/cmd_vel`` matches the ``/cmd_vel`` entry on the final-segment
    rule, so every command this robot sends is gated - which is what makes it
    the right instance to test the gate with. It is also the topic this
    module's own usage example drives.
    """
    return RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")


def _tool(robot: RtpsRobot, name: str) -> Any:
    return next(t for t in robot.tools if t.tool_name == name)


class TestRtpsCommandsReachTheGate:
    """The robot's agent tools must prompt the operator, not refuse outright."""

    writer: _FakeWriter
    bundle: _FakeBundle

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.writer, self.bundle = _install_doubles(monkeypatch)

    def test_drive_tool_prompts_the_operator_and_publishes_on_approval(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert ctx.interrupt.called, "the drive tool never reached the operator gate"
        assert ctx.interrupt.call_args[1]["reason"]["target"] == "/turtle1/cmd_vel"
        assert ctx.interrupt.call_args[1]["reason"]["action"] == "publish"
        assert result["status"] == "success"
        assert f"published 1 message(s) to /turtle1/cmd_vel ({_TWIST})" in _texts(result)
        assert len(self.writer.written) == 1

    def test_drive_tool_declined_publishes_nothing(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert result["status"] == "error"
        assert "declined by the operator" in _texts(result)
        assert self.writer.written == []

    def test_stop_tool_halts_the_robot_once_the_operator_approves(self) -> None:
        """The halt is gated like any other cmd_vel publish, and must be reachable.

        A transport that cannot forward an operator context makes ``stop`` an
        unconditional refusal - removing the one control the ``tools`` contract
        guarantees ("a caller that can start motion must be able to end it").
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "stop_turtlesim")(tool_context=ctx)
        assert ctx.interrupt.called, "the stop tool never reached the operator gate"
        assert result["status"] == "success"
        assert len(self.writer.written) == 1
        sample = self.writer.written[0]
        assert (sample.linear.x, sample.angular.z) == (0.0, 0.0)

    def test_the_trailing_zero_of_a_timed_drive_reaches_the_same_operator(self) -> None:
        """A timed drive owns its own stop, and that stop is gated too.

        The trailing zero is a second publish to the same blocklisted surface, so
        it needs its own approval. One that could not reach the gate would leave
        the robot latched at speed after the hold - the failure mode the fleet
        scope assertion in ``tests/mesh/test_drive_contract_fleet_scope.py``
        names, arriving through the operator path rather than the wire.
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel", publish_rate=2.0)
        result = _tool(robot, "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=ctx)
        assert result["status"] == "success"
        # Two prompts: one for the hold, one for the trailing zero that ends it.
        assert ctx.interrupt.call_count == 2
        # round(1.0 * 2.0) held samples, then a single zero.
        assert len(self.writer.written) == 3
        assert self.writer.written[0].linear.x == 1.0
        assert self.writer.written[-1].linear.x == 0.0

    def test_a_declined_hold_never_latches_a_velocity(self) -> None:
        """Refusing the hold refuses the motion, and the undo has nothing to undo.

        The trailing zero is skipped for a command that never moved the robot, so
        a declined drive writes no sample at all rather than a stop for a start
        that did not happen.
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel", publish_rate=2.0)
        result = _tool(robot, "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=ctx)
        assert result["status"] == "error"
        assert self.writer.written == []

    def test_programmatic_command_without_a_context_names_the_headless_variables(self) -> None:
        """The documented decision: no operator context means the gate refuses.

        A programmatic ``turtle.drive(...)`` has nobody to prompt, so the refusal
        has to say how an operator lifts it rather than reading as a broken API.
        This row is unchanged by whether the transport forwards, so it is the
        boundary control for the rows above rather than a check on the forward.
        """
        result = _turtle().drive(linear=1.0)
        assert result["status"] == "error"
        assert "STRANDS_ROS2_COMMAND_ALLOW" in _texts(result)
        assert "BYPASS_TOOL_CONSENT" in _texts(result)
        assert self.writer.written == []

    @pytest.mark.parametrize("allow", ["/turtle1/cmd_vel", "cmd_vel"])
    def test_programmatic_drive_and_stop_run_under_the_headless_allowlist(
        self, monkeypatch: pytest.MonkeyPatch, allow: str
    ) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", allow)
        robot = _turtle()
        assert robot.drive(linear=1.0)["status"] == "success"
        assert robot.stop()["status"] == "success"
        assert len(self.writer.written) == 2

    def test_advertise_is_not_gated_because_it_writes_no_sample(self) -> None:
        """Joining the graph as a publisher is not a command.

        ``advertise`` is RTPS-only and creates a writer without writing, so the
        gate deliberately does not cover it. Pinned here because an over-broad
        gate would make the one thing this transport can do that the bridge
        cannot unreachable without an operator.
        """
        result = _turtle().advertise()
        assert result["status"] == "success"
        assert self.writer.written == []


class TestTheDeclaredInterfaceReachesTheGatedPublish:
    """A gated publish must carry the interface the robot itself declares.

    The cases above stand the DDS backend and the IDL bundle down, so nothing
    there distinguishes the robot's declared ``cmd_vel_type`` from whatever the
    resolver hands back. ``use_rtps`` resolves the type, asks the backend for a
    writer on ``(topic, type)`` and names the type in its own success report, so
    all three are observable - and a declaration the bundle cannot carry has to
    be reported rather than published.
    """

    writer: _FakeWriter
    bundle: _FakeBundle

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.writer, self.bundle = _install_doubles(monkeypatch)

    def test_the_writer_is_opened_for_the_declared_type_on_the_robots_topic(self) -> None:
        """The transport passes the declared interface through, not only the topic."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert result["status"] == "success"
        assert self.bundle.resolved == [_TWIST]
        assert self.writer.requested == [("/turtle1/cmd_vel", _TWIST)]

    def test_an_interface_outside_the_bundle_writes_no_sample(self) -> None:
        """A declared type the bundle cannot resolve is reported, not published.

        The gate still runs first: the refusal belongs to the interface, so an
        operator who approved a command that could never be built is told why
        rather than being asked nothing and reading a success.
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        robot = RtpsRobot.from_rtps(
            node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel", cmd_vel_type=_OUTSIDE_THE_BUNDLE
        )
        result = _tool(robot, "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert ctx.interrupt.called, "the operator gate must run before the interface is resolved"
        assert result["status"] == "error"
        assert _OUTSIDE_THE_BUNDLE in _texts(result)
        # The robot's own declaration is what was offered to the resolver.
        assert self.bundle.resolved == [_OUTSIDE_THE_BUNDLE]
        assert self.writer.written == []

    def test_a_registered_interface_of_the_wrong_shape_writes_no_sample(self) -> None:
        """A type the bundle carries can still have nowhere to put a velocity.

        ``geometry_msgs/msg/Point`` passes the tool's ``pkg/msg/Name`` check and
        resolves, and has no ``linear``/``angular`` member - so the sample cannot
        be built and the wire stays quiet. This is the declaration a resolver
        that answered every type string would report as a completed drive.
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel", cmd_vel_type=_POINT)
        result = _tool(robot, "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert result["status"] == "error"
        assert "unknown field 'linear'" in _texts(result)
        assert self.bundle.resolved == [_POINT]
        # The sample cannot be built, so no writer ever joins the graph.
        assert self.writer.requested == []
        assert self.writer.written == []

    def test_the_default_declared_interface_is_the_one_the_doubles_carry(self) -> None:
        """Premise: the resolver double answers what an unconfigured robot declares."""
        default = inspect.signature(RtpsRobot).parameters["cmd_vel_type"].default
        assert default == _TWIST, f"the doubles carry {_TWIST!r} but a default robot declares {default!r}"

    def test_the_doubled_bundle_carries_only_types_the_real_bundle_carries(self) -> None:
        """Premise: the double stands in for real interfaces, not for a fiction.

        A faithful refusal is worth nothing if the types it does resolve are ones
        no real bundle has, or if the type used to provoke the refusal is one the
        real bundle would have resolved.
        """
        if not idl_mod.REGISTRY:
            pytest.skip("the RTPS IDL bundle needs cyclonedds (the [ros2] extra) to populate")
        invented = sorted(set(_BUNDLE) - set(idl_mod.REGISTRY))
        assert not invented, f"the doubled bundle carries types the real one does not: {invented}"
        assert _OUTSIDE_THE_BUNDLE not in idl_mod.REGISTRY, (
            f"{_OUTSIDE_THE_BUNDLE!r} is in the real bundle, so it cannot stand for a type outside it"
        )
