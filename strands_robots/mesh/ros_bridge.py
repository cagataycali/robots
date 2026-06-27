"""ROS 2 mesh bridge - present a remote ROS 2 robot as a strands Robot.

A :class:`RosBridgedRobot` wraps a ROS 2 mobile base (or any robot exposing a
``cmd_vel`` / odometry / scan topic trio) so that an agent can drive it and read
its state with the same ``Agent(tools=[robot])`` pattern used for simulated and
hardware robots. All ROS 2 I/O is forwarded through the
:func:`strands_robots.tools.use_ros.use_ros` tool, so the bridge stays thin and
inherits ``use_ros``'s in-process ``rclpy`` backend and input validation.

Typical usage::

    from strands import Agent
    from strands_robots.mesh import RosBridgedRobot

    turtle = RosBridgedRobot.from_ros(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
        odom_topic="/turtle1/pose",
        odom_type="turtlesim/msg/Pose",
    )

    # Direct, programmatic control:
    turtle.drive(linear=1.0)
    print(turtle.get_pose())

    # Or hand the bridge to an agent as first-class tools:
    agent = Agent(tools=turtle.tools)
    agent("drive forward, then tell me the pose")
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool
from strands.types.tools import AgentTool

from strands_robots.tools.use_ros import use_ros

_TWIST_TYPE = "geometry_msgs/msg/Twist"

# ROS 2 graph names: leading slash plus alnum / _ / ~ segments. Reject anything
# else early so a malformed topic fails at construction with a clear message
# rather than deep inside a forwarded ``use_ros`` call.
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_/~]+$")


def _check_topic(label: str, value: str) -> str:
    """Validate a ROS 2 topic/node name, returning it unchanged when valid."""
    if not value or not _TOPIC_RE.match(value):
        raise ValueError(f"invalid {label}: {value!r} (expected a ROS 2 graph name like /turtle1/cmd_vel)")
    return value


class RosBridgedRobot:
    """A remote ROS 2 robot exposed as a strands-controllable robot.

    The bridge owns no ROS 2 state of its own; every method forwards to
    :func:`use_ros`. It is therefore safe to construct without a ROS 2
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
            :meth:`get_scan`; when omitted, :meth:`get_scan` returns an error.
        cmd_vel_type: Interface type published to ``cmd_vel_topic``. Defaults to
            ``geometry_msgs/msg/Twist``.
        odom_type: Interface type of ``odom_topic``. Optional - when omitted,
            ``use_ros`` resolves it from the live graph.
        scan_type: Interface type of ``scan_topic``. Optional - resolved from
            the live graph when omitted.
        publish_rate: Default rate (Hz) for multi-message :meth:`drive` calls.
    """

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
    ) -> None:
        self.node_name = _check_topic("node_name", node_name)
        self.cmd_vel_topic = _check_topic("cmd_vel_topic", cmd_vel_topic)
        self.odom_topic = _check_topic("odom_topic", odom_topic)
        self.scan_topic = _check_topic("scan_topic", scan_topic) if scan_topic else None
        self.cmd_vel_type = cmd_vel_type
        self.odom_type = odom_type
        self.scan_type = scan_type
        self.publish_rate = publish_rate

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

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        """Publish a velocity command to the robot's ``cmd_vel`` topic.

        Args:
            linear: Forward linear velocity (m/s), mapped to ``linear.x``.
            angular: Yaw angular velocity (rad/s), mapped to ``angular.z``.
            duration: When given, hold the command for this many seconds by
                publishing ``round(duration * publish_rate)`` messages. Takes
                precedence over ``count``.
            count: Number of messages to publish when ``duration`` is omitted.

        Returns:
            The ``use_ros`` publish result dict.
        """
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        fields = {"linear": {"x": float(linear)}, "angular": {"z": float(angular)}}
        return use_ros(
            action="publish",
            topic=self.cmd_vel_topic,
            type=self.cmd_vel_type,
            fields=fields,
            count=n,
            rate=self.publish_rate,
        )

    def stop(self) -> dict[str, Any]:
        """Publish a zero-velocity command to halt the robot."""
        return self.drive(linear=0.0, angular=0.0, count=1)

    def get_pose(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's odometry/pose topic.

        Returns:
            The ``use_ros`` echo result dict for ``odom_topic``.
        """
        return use_ros(
            action="echo",
            topic=self.odom_topic,
            type=self.odom_type,
            count=1,
            timeout=timeout,
        )

    def get_scan(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's laser-scan topic.

        Returns:
            The ``use_ros`` echo result dict for ``scan_topic``, or an error
            result when no ``scan_topic`` was configured.
        """
        if not self.scan_topic:
            return {
                "status": "error",
                "content": [{"text": "get_scan: no scan_topic configured for this robot"}],
            }
        return use_ros(
            action="echo",
            topic=self.scan_topic,
            type=self.scan_type,
            count=1,
            timeout=timeout,
        )

    @property
    def tools(self) -> list[AgentTool]:
        """Return this robot's capabilities as named strands agent tools.

        The returned tools are bound to this instance and uniquely named with
        the ``node_name`` suffix so multiple bridged robots can coexist in a
        single ``Agent(tools=[...])`` call without name collisions.
        """
        suffix = self.node_name.strip("/").replace("/", "_")

        @tool(name=f"drive_{suffix}", description=f"Drive the {self.node_name} robot (linear/angular velocity).")
        def drive(linear: float = 0.0, angular: float = 0.0, duration: float | None = None) -> dict[str, Any]:
            return self.drive(linear=linear, angular=angular, duration=duration)

        @tool(name=f"get_pose_{suffix}", description=f"Read the current pose/odometry of the {self.node_name} robot.")
        def get_pose() -> dict[str, Any]:
            return self.get_pose()

        @tool(name=f"get_scan_{suffix}", description=f"Read one laser scan from the {self.node_name} robot.")
        def get_scan() -> dict[str, Any]:
            return self.get_scan()

        agent_tools: list[AgentTool] = [drive, get_pose]
        if self.scan_topic:
            agent_tools.append(get_scan)
        return agent_tools

    def __repr__(self) -> str:
        return (
            f"RosBridgedRobot(node_name={self.node_name!r}, cmd_vel_topic={self.cmd_vel_topic!r}, "
            f"odom_topic={self.odom_topic!r}, scan_topic={self.scan_topic!r})"
        )
