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
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.rtps.idl as idl_mod
import strands_robots.tools.use_rtps as rtps_mod
from strands_robots.mesh import RtpsRobot

_TWIST = "geometry_msgs/msg/Twist"


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


class _FakeWriter:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def write(self, sample: Any) -> None:
        self.written.append(sample)


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

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both env vars short-circuit the gate, so an ambient BYPASS_TOOL_CONSENT
        # (common in agent/automation shells) would make these assertions pass
        # without the gate ever running. Cases that need them opt in explicitly.
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
        self.writer = _FakeWriter()
        monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
        monkeypatch.setattr(rtps_mod._backend, "writer", lambda topic, type: self.writer)
        # publish sleeps for the reader settle and for the inter-message period.
        monkeypatch.setattr(rtps_mod.time, "sleep", lambda *_: None)
        monkeypatch.setattr(idl_mod, "get_type", lambda ros_type: _Twist)

    def test_drive_tool_prompts_the_operator_and_publishes_on_approval(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=ctx)
        assert ctx.interrupt.called, "the drive tool never reached the operator gate"
        assert ctx.interrupt.call_args[1]["reason"]["target"] == "/turtle1/cmd_vel"
        assert ctx.interrupt.call_args[1]["reason"]["action"] == "publish"
        assert result["status"] == "success"
        assert "published 1 message(s) to /turtle1/cmd_vel" in _texts(result)
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
