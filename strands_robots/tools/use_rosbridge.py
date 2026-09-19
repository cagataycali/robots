#!/usr/bin/env python3
"""Universal rosbridge tool - any ROS graph over a WebSocket, no ROS install.

Where :func:`strands_robots.tools.use_ros.use_ros` speaks to a ROS 2 graph
through in-process ``rclpy`` (requiring a sourced distro), ``use_rosbridge``
speaks the rosbridge JSON protocol over a WebSocket via ``roslibpy`` - a pure
pip dependency. That gives it two properties no other transport here has:

* **ROS 1 robots** (rosbridge_suite ships for ROS1 and ROS2 alike) - e.g. the
  NASA Curiosity rover Gazebo simulation (ROS1 Noetic).
* **No ROS environment on this machine** - the agent can run on macOS, CI, or
  any laptop and drive a robot across the network.

Requirements:
    ``pip install "strands-robots[rosbridge]"`` (roslibpy). The robot side
    runs ``rosbridge_server`` (with ``rosapi``) - standard in every rosbridge
    install. rosbridge is unauthenticated by default: use on trusted networks.

Graph introspection uses the ``rosapi`` node's services. Interface types are
ROS1-style two-segment names (``geometry_msgs/Twist``); field payloads are
plain JSON dicts, exactly as rosbridge transmits them.

This module is the agent-facing envelope: the numeric-option domains an agent
can get wrong, the operator gate, and the tool docstring a model reads. The
transport itself - the long-lived ``roslibpy.Ros`` per ``(host, port)``, the
``rosapi`` graph introspection, the host / port / name domains and the action
dispatch - is :mod:`strands_robots.rosbridge`, which
:class:`~strands_robots.mesh.rosbridge_robot.RosbridgeRobot` forwards through
as well.

Actions:
    status         - roslibpy availability + connectivity to host:port.
    list_topics    - every topic rosapi reports, with its type where rosapi
                     reports one (rosapi /rosapi/topics).
    list_services  - services (rosapi /rosapi/services).
    echo           - subscribe and return up to N messages as JSON. Type
                     auto-resolved via rosapi when omitted.
    publish        - advertise, publish N messages built from ``fields``,
                     unadvertise.
    service_call   - call a service with a JSON request dict.

Examples:
    use_rosbridge(action="status", host="192.168.1.20")
    use_rosbridge(action="list_topics")
    use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=1)
    use_rosbridge(action="publish",
                  topic="/curiosity_mars_rover/ackermann_drive_controller/cmd_vel",
                  type="geometry_msgs/Twist",
                  fields={"linear": {"x": 1.0}, "angular": {"z": 0.0}})
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots._command_gate import gate_command
from strands_robots.rosbridge import GATE_TOOL, _err, never_gated, rosbridge_action
from strands_robots.tools._numeric_options import numeric_option_error

# Which numeric options each action actually consumes. Unlike the rclpy
# transport, whose graph introspection reads no caller budget, EVERY action here
# begins with a timeout-bounded WebSocket dial, so every action reads
# ``timeout``. The guard below is driven by this table rather than validating
# the whole signature unconditionally, so a caller is never refused for a value
# the requested action never looks at.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "status": ("timeout",),
    "list_topics": ("timeout",),
    "list_services": ("timeout",),
    "echo": ("timeout", "count"),
    "service_call": ("timeout",),
    "publish": ("timeout", "count", "rate"),
}


@tool(context=True)
def use_rosbridge(
    action: str,
    host: str = "localhost",
    port: int = 9090,
    topic: str | None = None,
    service: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Universal rosbridge tool - ROS over a WebSocket, no ROS install needed.

    Args:
        action: One of ``status``, ``list_topics``, ``list_services``,
            ``echo``, ``publish``, ``service_call``.
        host: rosbridge server hostname or IP. Held to the shared domain every
            dialled host in this package shares, then to this transport's own
            narrower allowlist.
        port: rosbridge WebSocket port (default 9090).
        topic: Topic name (``echo``, ``publish``). Held to the same name rule
            as ``service``.
        service: Service name (``service_call``). Held to the same name rule
            as ``topic``.
        type: ROS1 two-segment interface type, e.g. ``geometry_msgs/Twist``.
            Required for ``publish`` - a message cannot be built without it -
            and auto-resolved for ``echo`` when omitted.
        fields: JSON field dict (``publish`` message / ``service_call`` request).
        timeout: Seconds for the WebSocket dial, sample collection, or a
            service call. A positive finite number of seconds; every action
            dials the bridge, so every action reads it.
        count: Messages to echo or publish. A positive integer; it is consumed
            as a ``range()`` bound, so ``0`` publishes nothing and a float or a
            numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.
        tool_context: Injected agent context, used to ask an operator before a
            ``publish`` or ``service_call`` reaches a safety-critical command surface.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    # Numeric options are checked here, ahead of the transport's own domains and
    # its backend probe, so the same caller mistake is reported identically
    # whether or not roslibpy is installed - and so a refusal happens before the
    # WebSocket is dialed and before a publisher is advertised. They are the
    # agent-supplied half of the call, which is why the table above lives beside
    # this tool rather than in the transport.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    # Each verb forwards the options it reads and no others, and only the two
    # verbs that carry a command are handed a gate that can reach an operator:
    # the read paths cannot prompt at all, rather than being trusted not to. An
    # action outside this vocabulary falls through to the transport, which names
    # it in its refusal.
    address: dict[str, Any] = {"host": host, "port": port, "timeout": timeout}
    if action == "publish":
        return rosbridge_action(
            action=action,
            topic=topic,
            type=type,
            fields=fields,
            count=count,
            rate=rate,
            gate=lambda kind, target: gate_command(kind, target, tool_context=tool_context, tool=GATE_TOOL),
            **address,
        )
    if action == "service_call":
        return rosbridge_action(
            action=action,
            service=service,
            type=type,
            fields=fields,
            gate=lambda kind, target: gate_command(kind, target, tool_context=tool_context, tool=GATE_TOOL),
            **address,
        )
    if action == "echo":
        return rosbridge_action(action=action, topic=topic, type=type, count=count, gate=never_gated, **address)
    if action in ("status", "list_topics", "list_services"):
        return rosbridge_action(action=action, gate=never_gated, **address)
    return rosbridge_action(action=action, topic=topic, service=service, type=type, gate=never_gated, **address)


__all__ = ["use_rosbridge"]
