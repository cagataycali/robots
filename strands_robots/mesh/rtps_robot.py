"""RTPS mesh robot - act as (or drive) a ROS 2 robot with no rclpy.

:class:`RtpsRobot` is the pure-RTPS sibling of :class:`RosBridgedRobot`. Where
the ROS bridge forwards to ``use_ros`` (which needs a sourced ROS 2 distro),
``RtpsRobot`` forwards to ``use_rtps`` - a DDS participant built on the
pip-installable ``cyclonedds`` binding alone. It therefore works on macOS,
Jetson, and CI with nothing but a pip wheel, and interoperates with every ROS 2
distro over RTPS.

Because an RTPS participant publishes real DDS samples, an :class:`RtpsRobot`
can do something the client-only bridge cannot: **act as a robot**. Advertise a
``cmd_vel`` listener or publish ``/odom`` / ``/joint_states`` and a real ROS 2
stack (rviz, nav2) treats the agent as hardware.

Typical usage::

    from strands import Agent
    from strands_robots.mesh import RtpsRobot

    turtle = RtpsRobot.from_rtps(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
    )
    turtle.drive(linear=1.0, duration=1.5)   # publishes Twist over RTPS
    agent = Agent(tools=turtle.tools)
    agent("drive forward for two seconds")

Scope mirrors ``use_rtps``: topics only, and types bounded by the IDL bundle
(``geometry_msgs/msg/Twist`` for ``drive``). Pose/scan read-back needs those
messages in the bundle; until they are added, use ``RosBridgedRobot`` for echo.
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool
from strands.types.tools import AgentTool

from strands_robots.tools.use_rtps import use_rtps

_TWIST_TYPE = "geometry_msgs/msg/Twist"
_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_/]*[A-Za-z0-9_]$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~]+$")


def _check(label: str, value: str, pattern: re.Pattern[str]) -> str:
    if not value or not pattern.match(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


class RtpsRobot:
    """A ROS 2 robot driven over pure RTPS (no rclpy), exposed as a strands robot.

    The robot owns no DDS state of its own; every method forwards to
    :func:`use_rtps`, which manages the shared participant and cached writers.
    Safe to construct without cyclonedds present - errors surface only when a
    method is called and the backend is unavailable.

    Args:
        node_name: Identifier used to name this robot's agent tools
            (``drive_<node_name>`` etc.). Need not match any ROS 2 node name.
        cmd_vel_topic: Velocity-command topic to publish ``Twist`` on.
        cmd_vel_type: Interface type for ``cmd_vel_topic`` (default
            ``geometry_msgs/msg/Twist``).
        publish_rate: Default rate (Hz) for multi-message :meth:`drive` calls.
    """

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        *,
        cmd_vel_type: str = _TWIST_TYPE,
        publish_rate: float = 10.0,
    ) -> None:
        self.node_name = _check("node_name", node_name, _NAME_RE)
        self.cmd_vel_topic = _check("cmd_vel_topic", cmd_vel_topic, _TOPIC_RE)
        self.cmd_vel_type = cmd_vel_type
        self.publish_rate = publish_rate

    @classmethod
    def from_rtps(
        cls,
        node_name: str,
        cmd_vel_topic: str,
        **kwargs: Any,
    ) -> RtpsRobot:
        """Construct an RTPS robot from ROS 2 topic wiring.

        Keyword-style alternate constructor mirroring
        :meth:`RosBridgedRobot.from_ros`. The top-level ``Robot`` is a factory
        *function* (not a class), so the alternate constructor lives here where
        it is type-safe and discoverable.
        """
        return cls(node_name, cmd_vel_topic, **kwargs)

    def advertise(self) -> dict[str, Any]:
        """Create the ``cmd_vel`` publisher up front (appear on the ROS 2 graph)."""
        return use_rtps(action="advertise", topic=self.cmd_vel_topic, type=self.cmd_vel_type)

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        """Publish a velocity command over RTPS to the robot's ``cmd_vel`` topic.

        Args:
            linear: Forward linear velocity (m/s), mapped to ``linear.x``.
            angular: Yaw angular velocity (rad/s), mapped to ``angular.z``.
            duration: When given, hold the command for this many seconds
                (publishes ``round(duration * publish_rate)`` messages).
            count: Message count when ``duration`` is omitted.
        """
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        fields = {"linear": {"x": float(linear)}, "angular": {"z": float(angular)}}
        return use_rtps(
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

    @property
    def tools(self) -> list[AgentTool]:
        """Return this robot's capabilities as uniquely-named strands agent tools."""
        suffix = self.node_name.strip("/").replace("/", "_")

        @tool(
            name=f"drive_{suffix}", description=f"Drive the {self.node_name} robot over RTPS (linear/angular velocity)."
        )
        def drive(linear: float = 0.0, angular: float = 0.0, duration: float | None = None) -> dict[str, Any]:
            return self.drive(linear=linear, angular=angular, duration=duration)

        @tool(name=f"stop_{suffix}", description=f"Stop the {self.node_name} robot (zero velocity).")
        def stop() -> dict[str, Any]:
            return self.stop()

        return [drive, stop]

    def __repr__(self) -> str:
        return (
            f"RtpsRobot(node_name={self.node_name!r}, cmd_vel_topic={self.cmd_vel_topic!r}, "
            f"cmd_vel_type={self.cmd_vel_type!r})"
        )
