"""ROS 2 mesh bridge - present a remote ROS 2 robot as a strands Robot.

A :class:`RosBridgedRobot` wraps a ROS 2 mobile base (or any robot exposing a
``cmd_vel`` / odometry / scan topic trio) so that an agent can drive it and read
its state with the same ``Agent(tools=[robot])`` pattern used for simulated and
hardware robots. All ROS 2 I/O is forwarded through
:func:`strands_robots.ros.ros_action` - the same transport the ``use_ros`` tool
is an agent envelope over - so the bridge stays thin and inherits the in-process
``rclpy`` backend and its topic/type validation. The parameters the transport
never sees are validated here: a
:meth:`RosBridgedRobot.drive` command whose velocity, hold duration or message
count cannot be honored is refused without publishing anything, and a
:meth:`RosBridgedRobot.navigate_to` goal whose pose cannot be honored is refused
without sending anything - the goal coordinates travel inside the request body,
which the transport forwards verbatim.

The drive contract, its safety semantics, and the ``tools`` property live in
:class:`~strands_robots.mesh._mobile_base.MobileBaseRobot`; this module supplies
the ``rclpy`` transport and the ROS 2-specific Nav2 goal surface.

Commanding a robot goes through the shared operator-approval gate, because
``/turtle1/cmd_vel`` and a Nav2 ``/navigate_to_pose`` are safety-critical
surfaces. The bridge therefore forwards an operator context to it: the
``drive_<node>`` / ``stop_<node>`` / ``navigate_<node>`` agent tools are declared
``@tool(context=True)`` and hand the context to the gate, so an agent driving
this robot prompts the operator exactly as a direct ``use_ros`` call does - the
label is the transport's own :data:`~strands_robots.ros.GATE_TOOL`, so the same
physical topic files one interrupt id and one audit source whichever surface
reached it. ``drive`` and ``stop`` are declared by the shared base and reach this
module's transport, which is the single place the context is handed to the gate;
``navigate`` is declared here because the Nav2 goal is ROS 2-only. The read paths
(:meth:`get_pose`, :meth:`get_scan`) are never gated.

A **programmatic** call carries no operator context, so it is refused unless the
surface is pre-approved with ``STRANDS_ROS2_COMMAND_ALLOW`` (or the gate is
bypassed with ``BYPASS_TOOL_CONSENT=true``). That includes :meth:`stop`: the gate
is keyed on the *surface*, not on the payload, so a zero-velocity halt is gated
like any other publish to ``cmd_vel``. A payload-shaped exemption would have to
know that zero means "stationary" on a ``Twist`` while zero on ``/joint_command``
commands motion to the zero pose - so the halt stays reachable through the same
three approval paths rather than through a carve-out.

Typical usage::

    import os

    from strands import Agent
    from strands_robots.mesh import RosBridgedRobot

    turtle = RosBridgedRobot.from_ros(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
        odom_topic="/turtle1/pose",
        odom_type="turtlesim/msg/Pose",
    )

    # Direct, programmatic control - no operator context to prompt, so the
    # command surface has to be pre-approved:
    os.environ["STRANDS_ROS2_COMMAND_ALLOW"] = "/turtle1/cmd_vel"
    turtle.drive(linear=1.0)
    print(turtle.get_pose())  # reads are never gated

    # Or hand the bridge to an agent as first-class tools - the command tools
    # carry the operator context, so the gate prompts instead of refusing:
    agent = Agent(tools=turtle.tools)
    agent("drive forward, then tell me the pose")
"""

from __future__ import annotations

import math
import re
from typing import Any, cast

from strands import tool
from strands.types.tools import AgentTool, ToolContext

from strands_robots._command_gate import gate_command
from strands_robots.mesh._mobile_base import ActionCapable, MobileBaseRobot
from strands_robots.ros import GATE_TOOL, CommandGate, never_gated, ros_action
from strands_robots.utils import (
    finite_number_error,
    positive_finite_number_error,
)

_TWIST_TYPE = "geometry_msgs/msg/Twist"
_NAV_ACTION_TYPE = "nav2_msgs/action/NavigateToPose"

# ROS 2 graph names: leading slash plus alnum / _ / ~ segments. Reject anything
# else early so a malformed topic fails at construction with a clear message
# rather than deep inside a forwarded transport call.
_ROS2_GRAPH_NAME_RE = re.compile(r"^[A-Za-z0-9_/~]+\Z")


def _check_topic(label: str, value: str) -> str:
    """Validate a ROS 2 topic/node name, returning it unchanged when valid.

    Retained as a module-level function because it is the validator other ROS-
    flavored bridges reach for. Prefer :meth:`RosBridgedRobot._check_topic`,
    which picks up a subclass's own grammar.
    """
    if not value or not _ROS2_GRAPH_NAME_RE.match(value):
        raise ValueError(f"invalid {label}: {value!r} (expected a ROS 2 graph name like /turtle1/cmd_vel)")
    return value


def _operator_gate(tool_context: ToolContext | None) -> CommandGate:
    """Return the gate a command verb consults, keyed on the shared tool label.

    The label builds the interrupt id ``<tool>-command-approval`` and the audit
    source ``<tool>_tool``, so a bridge that spelled it differently from the
    ``use_ros`` tool would file one incident's rows under two names and ask an
    operator the same question twice. Both ROS 2 mesh bridges call this.

    Args:
        tool_context: The operator context an agent tool was invoked with, or
            ``None`` for a programmatic caller - which the gate then refuses
            unless the surface is pre-approved.

    Returns:
        A :data:`~strands_robots.ros.CommandGate` closure over the context.
    """
    return lambda kind, target: gate_command(kind, target, tool_context=tool_context, tool=GATE_TOOL)


class _UseRosTransport:
    """Transport that forwards to the in-process ``rclpy`` ROS 2 transport.

    Every method resolves :func:`~strands_robots.ros.ros_action` through this
    module's globals rather than capturing it at import, so tests (and any
    operator patching the transport) can monkeypatch
    ``strands_robots.mesh.ros_bridge.ros_action`` and have the bridge honor it.
    Implements the full optional surface: ROS 2 has services and actions.

    Every command verb hands the transport an operator gate built from
    ``tool_context``, which refuses a safety-critical surface it cannot get a
    decision for. This class is the one place that hand-off happens for the ROS 2
    bridge - the base carries the context down to the transport and no further.
    ``echo`` is handed :func:`~strands_robots.ros.never_gated`: a read is never
    gated.
    """

    twist_type = _TWIST_TYPE

    def publish(
        self,
        *,
        topic: str,
        type: str,
        fields: dict[str, Any],
        count: int,
        rate: float,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        return ros_action(
            action="publish",
            topic=topic,
            type=type,
            fields=fields,
            count=count,
            rate=rate,
            gate=_operator_gate(tool_context),
        )

    def echo(self, *, topic: str, type: str | None, count: int, timeout: float) -> dict[str, Any]:
        return ros_action(action="echo", topic=topic, type=type, count=count, timeout=timeout, gate=never_gated)

    def service_call(
        self,
        *,
        service: str,
        type: str,
        fields: dict[str, Any],
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        return ros_action(
            action="service_call", service=service, type=type, fields=fields, gate=_operator_gate(tool_context)
        )

    def action_send_goal(
        self,
        *,
        action_name: str,
        type: str,
        fields: dict[str, Any],
        timeout: float,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        return ros_action(
            action="action_send_goal",
            action_name=action_name,
            type=type,
            fields=fields,
            timeout=timeout,
            gate=_operator_gate(tool_context),
        )


class RosBridgedRobot(MobileBaseRobot):
    """A remote ROS 2 robot exposed as a strands-controllable robot.

    The bridge owns no ROS 2 state of its own; every method forwards to
    :func:`~strands_robots.ros.ros_action`. It is therefore safe to construct
    without a ROS 2
    environment present - errors surface only when a method is actually called
    and no backend is available.

    Args:
        node_name: Human-readable identifier for the remote robot. Used only to
            name this bridge's agent tools (``drive_<node_name>`` etc.); it does
            not need to match the ROS 2 node name.
        cmd_vel_topic: Velocity-command topic the robot subscribes to (e.g.
            ``/turtle1/cmd_vel`` or ``/cmd_vel``).
        odom_topic: Topic carrying the robot's pose/odometry (e.g.
            ``/turtle1/pose`` or ``/odom``). Read by :meth:`get_pose`.
        scan_topic: Optional laser-scan topic (e.g. ``/scan``). Read by
            :meth:`get_scan`; when omitted, no ``get_scan`` tool is exposed.
        cmd_vel_type: Interface type published to ``cmd_vel_topic``. Defaults to
            ``geometry_msgs/msg/Twist``.
        odom_type: Interface type of ``odom_topic``. Optional - when omitted,
            the transport resolves it from the live graph.
        scan_type: Interface type of ``scan_topic``. Optional - resolved from
            the live graph when omitted.
        publish_rate: Default rate (Hz) for multi-message :meth:`drive` calls.
            Must be > 0 and finite: :meth:`drive` multiplies it by ``duration``
            to size the message burst and the transport publishes at ``1 / rate``,
            so a non-positive rate removes the pacing entirely rather than
            slowing it. Raises ``ValueError`` at construction otherwise.
        max_linear: Optional linear-velocity clamp (m/s). Unset by default: a
            generic ROS 2 base declares no speed limit to this bridge, and
            inventing one would silently cap an existing caller.
        max_angular: Optional angular-velocity clamp (rad/s). Unset by default.
        max_duration: Optional cap on a single :meth:`drive` hold, in seconds.
            Unset by default; when set, a longer request is refused loudly.
        nav_action: Optional Nav2-style action server name (e.g.
            ``/navigate_to_pose``). When set, :meth:`navigate_to` sends
            goal-level navigation instead of raw velocity, and a
            ``navigate_<node_name>`` agent tool is exposed.
        nav_action_type: Action interface for ``nav_action``. Defaults to
            ``nav2_msgs/action/NavigateToPose``.
    """

    # ROS 2 uses one grammar for node names and topics alike, so both seams
    # point at the same pattern. Bound through a differently-named module global
    # because `_TOPIC_RE = _TOPIC_RE` in a class body reads as a typo.
    _NAME_RE = _ROS2_GRAPH_NAME_RE
    _TOPIC_RE = _ROS2_GRAPH_NAME_RE
    _NAME_HINT = " (expected a ROS 2 graph name like /turtle1/cmd_vel)"
    # Both seams share one grammar here, so one sentence describes both.
    _TOPIC_HINT = _NAME_HINT

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        odom_topic: str,
        scan_topic: str | None = None,
        *,
        cmd_vel_type: str = _TWIST_TYPE,
        odom_type: str | None = None,
        scan_type: str | None = None,
        publish_rate: float = 10.0,
        max_linear: float | None = None,
        max_angular: float | None = None,
        max_duration: float | None = None,
        nav_action: str | None = None,
        nav_action_type: str = _NAV_ACTION_TYPE,
    ) -> None:
        # ``odom_topic`` is optional on the base (the DeepRacer has no
        # odometry at all) but required here, so an empty value is a malformed
        # argument rather than an omission and must be refused, not read as
        # "no odometry wired".
        self._check_topic("odom_topic", odom_topic)
        super().__init__(
            node_name,
            cmd_vel_topic,
            _UseRosTransport(),
            odom_topic=odom_topic,
            scan_topic=scan_topic,
            cmd_vel_type=cmd_vel_type,
            odom_type=odom_type,
            scan_type=scan_type,
            max_linear=max_linear,
            max_angular=max_angular,
            max_duration=max_duration,
            publish_rate=publish_rate,
        )
        self.nav_action = self._check_topic("nav_action", nav_action) if nav_action else None
        self.nav_action_type = nav_action_type

    @classmethod
    def from_ros(
        cls,
        node_name: str,
        cmd_vel_topic: str,
        odom_topic: str,
        scan_topic: str | None = None,
        **kwargs: Any,
    ) -> RosBridgedRobot:
        """Construct a bridge from ROS 2 topic wiring.

        Convenience alternate constructor mirroring the keyword style used
        elsewhere in the library. Equivalent to calling the constructor
        directly; provided so call sites read as ``RosBridgedRobot.from_ros(
        node_name=..., cmd_vel_topic=...)``.
        """
        return cls(node_name, cmd_vel_topic, odom_topic, scan_topic, **kwargs)

    def navigate_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        timeout: float = 120.0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Send a goal-level navigation request to the robot's ``nav_action``.

        Unlike :meth:`drive`, which streams raw velocity, this delegates
        obstacle avoidance, path planning, and recovery to the robot's own
        navigation stack (Nav2 by default) and blocks until the goal reaches a
        terminal state or ``timeout`` expires - at which point the transport
        cancels the goal so the robot does not keep navigating unattended.

        Args:
            x: Goal position x in ``frame_id`` (meters). Must be a finite
                number; both signs are valid.
            y: Goal position y in ``frame_id`` (meters). Must be a finite
                number; both signs are valid.
            yaw: Goal heading in radians, encoded as a planar quaternion. Must
                be a finite number; both signs are valid (negative turns the
                other way).
            frame_id: Frame the goal pose is expressed in (default ``map``).
            timeout: End-to-end budget in seconds for the navigation goal.
                Graded here, on the domain :meth:`drive` grades ``duration``
                against: a non-positive or non-finite budget is refused and no
                goal is sent.
            tool_context: Operator context forwarded to the command gate, which
                covers a Nav2-style ``/navigate_to_pose`` action goal as well as
                a ``cmd_vel`` publish (see :meth:`drive`).

        Returns:
            The transport's action result dict (goal status, result, feedback
            samples), or an ``{"status": "error"}`` result when no
            ``nav_action`` was configured or when a pose component cannot be
            honored - in which case no goal is sent.
        """
        if not self.nav_action:
            return self._error("navigate_to: no nav_action configured for this robot")
        # The goal pose is the part of this call the transport never validates: it
        # checks the action name and interface type, but the coordinates travel
        # inside ``fields`` and are serialized into the request verbatim. A
        # non-finite coordinate is a valid IEEE-754 float64 on the wire, so the
        # goal is accepted and handed to a planner that cannot resolve it, and
        # ``yaw`` additionally reaches ``math.sin``/``math.cos``, which raise a
        # bare ``ValueError`` for an infinite angle - out of a method whose
        # contract is a result dict, and out of the bound ``navigate_*`` tool.
        # ``timeout`` is graded here for the same reason ``get_pose`` grades its
        # wait: a transport reached by more than one surface honors the budget it
        # is handed and states no domain of its own, and an ungraded ``inf``
        # becomes an unbounded ``wait_for_server`` held inside the backend's
        # process-wide lock. This stays on the subclass because ``nav_action`` is
        # a ROS 2 concept: no other transport has a goal-level navigation
        # surface to guard.
        goal_error = (
            finite_number_error(x, "x", "navigate_to")
            or finite_number_error(y, "y", "navigate_to")
            or finite_number_error(yaw, "yaw", "navigate_to")
            or positive_finite_number_error(timeout, "timeout", "navigate_to")
        )
        if goal_error:
            return self._error(goal_error)
        half = 0.5 * float(yaw)
        fields = {
            "pose": {
                "header": {"frame_id": frame_id},
                "pose": {
                    "position": {"x": float(x), "y": float(y)},
                    "orientation": {"z": math.sin(half), "w": math.cos(half)},
                },
            }
        }
        return cast(ActionCapable, self.transport).action_send_goal(
            action_name=self.nav_action,
            type=self.nav_action_type,
            fields=fields,
            timeout=timeout,
            tool_context=tool_context,
        )

    def _extra_tools(self) -> list[AgentTool]:
        """Expose ``navigate_<suffix>`` only when a nav action is wired."""
        if not self.nav_action:
            return []
        suffix = self.tool_suffix

        @tool(
            name=f"navigate_{suffix}",
            description=(
                f"Navigate the {self.node_name} robot to a map-frame (x, y, yaw) goal using its "
                "navigation stack (planning and obstacle avoidance handled on-robot)."
            ),
            context=True,
        )
        def navigate(
            x: float,
            y: float,
            yaw: float = 0.0,
            timeout: float = 120.0,
            tool_context: ToolContext | None = None,
        ) -> dict[str, Any]:
            return self.navigate_to(x=x, y=y, yaw=yaw, timeout=timeout, tool_context=tool_context)

        return [navigate]
