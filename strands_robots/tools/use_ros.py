#!/usr/bin/env python3
"""Universal ROS 2 bridge tool - one tool for the full ROS 2 surface.

Like ``use_lerobot`` wraps the lerobot module tree, ``use_ros`` gives a Strands
agent a single, structured entry point into any ROS 2 graph reachable from this
interpreter - **entirely in-process through ``rclpy``**. There is no shelling
out to the ``ros2`` CLI and no code-generation: every action calls the ROS 2
client library directly, so message types are real Python classes, errors are
real exceptions, and a single long-lived node/executor is reused across calls.

This module is the agent-facing envelope: the numeric-option domains an agent
can get wrong, the operator gate, and the tool docstring a model reads. The
transport itself - the process-wide ``rclpy`` node and executor, the dynamic
type resolution, the graph introspection and the action-goal lifecycle - is
:mod:`strands_robots.ros`, which
:class:`~strands_robots.mesh.ros_bridge.RosBridgedRobot` and
:class:`~strands_robots.mesh.ackermann_robot.AckermannRosRobot` forward through
as well.

Requirements:
    ``rclpy`` and ``rosidl_runtime_py`` must be importable in this interpreter.
    These ship with a sourced system ROS 2 distro (apt / RoboStack / conda) and
    are **not** on PyPI, so they cannot be ``pip install``ed and are not pinned
    in ``pyproject.toml`` (the ``[ros2]`` extra only carries the pip-installable
    ``cyclonedds`` RMW binding). Source a ROS 2 environment before launching the
    agent - e.g. ``source /opt/ros/jazzy/setup.bash`` - and ``rclpy`` becomes
    importable. When it is absent, every action returns a clear, actionable
    error instead of raising.

Message and service types are resolved dynamically through
``rosidl_runtime_py`` (``get_message`` / ``get_service``), so any interface
installed in the ROS 2 environment works with no static registry. Field
payloads are passed as plain JSON dicts and applied with ``set_message_fields``
- the standard ROS 2 idiom.

Actions:
    status         - report whether the in-process rclpy backend is available.
    list_topics    - list topics with their types.
    list_nodes     - list nodes.
    list_services  - list services with their types.
    info           - describe a topic (type + pub/sub counts) or service (type).
    echo           - subscribe to a topic and return N samples as JSON.
    publish        - publish N messages built from a JSON field dict.
    service_call   - call a service with a JSON request dict, return the response.
    list_actions   - list action servers with their types.
    action_send_goal - send a goal to an action server, stream feedback, and
                     return the terminal result (goal-level autonomy: Nav2
                     NavigateToPose, FollowJointTrajectory, gripper commands).
                     On timeout the goal is cancelled before returning, so a
                     robot is never left executing an orphaned goal.

Examples:
    use_ros(action="status")
    use_ros(action="list_topics")
    use_ros(action="echo", topic="/turtle1/pose", timeout=2.0, count=2)
    use_ros(action="publish", topic="/turtle1/cmd_vel",
            type="geometry_msgs/msg/Twist",
            fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})
    use_ros(action="service_call", service="/spawn",
            type="turtlesim/srv/Spawn",
            fields={"x": 3.0, "y": 3.0, "name": "t2"})
    use_ros(action="action_send_goal", action_name="/navigate_to_pose",
            type="nav2_msgs/action/NavigateToPose",
            fields={"pose": {"header": {"frame_id": "map"},
                             "pose": {"position": {"x": 1.0, "y": 2.0}}}},
            timeout=120.0)
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots._command_gate import gate_command
from strands_robots.ros import GATE_TOOL, _err, never_gated, ros_action
from strands_robots.tools._numeric_options import numeric_option_error

# Which numeric options each action actually consumes. An action that reads none
# of them (``status``, the ``list_*`` queries, ``info``) must not be refused for
# a value it never looks at, so the guard below is driven by this table rather
# than validating the whole signature unconditionally.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "echo": ("timeout", "count"),
    "publish": ("count", "rate"),
    "service_call": ("timeout",),
    "action_send_goal": ("timeout",),
}


@tool(context=True)
def use_ros(
    action: str,
    tool_context: ToolContext | None = None,
    topic: str | None = None,
    service: str | None = None,
    action_name: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
) -> dict[str, Any]:
    """Universal ROS 2 tool - in-process rclpy, dynamic types, no shelling out.

    Args:
        action: One of ``status``, ``list_topics``, ``list_nodes``,
            ``list_services``, ``list_actions``, ``info``, ``echo``,
            ``publish``, ``service_call``, ``action_send_goal``.
        tool_context: Injected agent context, used to ask an operator before a
            ``publish``, ``service_call`` or ``action_send_goal`` reaches a
            safety-critical command surface.
        topic: Topic name (``echo``, ``publish``, ``info``).
        service: Service name (``service_call``, ``info``).
        action_name: Action server name (``action_send_goal``), e.g.
            ``/navigate_to_pose``.
        type: Fully-qualified interface type, e.g. ``geometry_msgs/msg/Twist``,
            ``turtlesim/srv/Spawn``, or ``nav2_msgs/action/NavigateToPose``.
            Auto-resolved for ``echo`` when omitted.
        fields: JSON field dict applied with ``set_message_fields`` (``publish``,
            ``service_call``, ``action_send_goal``). Booleans and nulls are
            preserved - the dict is passed straight to rclpy, never serialised
            through source.
        timeout: Seconds to wait for samples / a service / an action result.
            For ``action_send_goal`` this is the end-to-end budget (discovery +
            acceptance + execution); size it to the goal (e.g. 120 for a Nav2
            navigation), and note the goal is cancelled when it expires.
            A positive finite number of seconds.
        count: Number of messages to echo or publish. A positive integer;
            it is consumed as a ``range()`` bound, so ``0`` publishes nothing
            and a float or a numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    # Numeric options are checked here, ahead of the transport's own name
    # validation and its backend probe, so the same caller mistake is reported
    # identically whether or not rclpy is installed - and so a refusal happens
    # before a publisher joins the graph. They are the agent-supplied half of the
    # call, which is why the table above lives beside this tool rather than in
    # the transport.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    # Each verb forwards the options it reads and no others, and only the three
    # verbs that carry a command are handed a gate that can reach an operator:
    # the read paths cannot prompt at all, rather than being trusted not to. An
    # action outside this vocabulary falls through to the transport, which names
    # it in its refusal.
    def operator_gate(kind: str, target: str) -> str | None:
        return gate_command(kind, target, tool_context=tool_context, tool=GATE_TOOL)

    if action == "publish":
        return ros_action(
            action=action, topic=topic, type=type, fields=fields, count=count, rate=rate, gate=operator_gate
        )
    if action == "service_call":
        return ros_action(action=action, service=service, type=type, fields=fields, timeout=timeout, gate=operator_gate)
    if action == "action_send_goal":
        return ros_action(
            action=action,
            action_name=action_name,
            type=type,
            fields=fields,
            timeout=timeout,
            gate=operator_gate,
        )
    if action == "echo":
        return ros_action(action=action, topic=topic, type=type, timeout=timeout, count=count, gate=never_gated)
    if action == "info":
        return ros_action(action=action, topic=topic, service=service, gate=never_gated)
    if action in ("status", "list_topics", "list_nodes", "list_services", "list_actions"):
        return ros_action(action=action, gate=never_gated)
    return ros_action(action=action, topic=topic, service=service, action_name=action_name, type=type, gate=never_gated)


__all__ = ["use_ros"]
