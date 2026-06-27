"""Publish hardware-robot telemetry on a ROS 2 domain.

When a :class:`strands_robots.hardware_robot.Robot` is constructed with
``ros2_bridge=True``, it owns a :class:`HardwareRosBridge` that advertises the
real arm's live observation on a ROS 2 domain. Any ROS 2 node on that domain can
then ``ros2 topic echo /<robot>/joint_states`` (or subscribe to the camera
``image_raw`` topics) against the physical robot, and the agent's own
``use_ros`` calls reach the same graph - the real device becomes a first-class
ROS 2 participant with no extra ROS nodes to launch.

:class:`HardwareRosBridge` is the hardware half of a symmetric pair: the
simulation half is :class:`strands_robots.simulation.ros_bridge.SimRosBridge`.
Both are thin subclasses of
:class:`strands_robots.ros_telemetry.RosTelemetryBridge` and publish the
identical per-robot topics, so a real arm and its digital twin are
indistinguishable on the ROS 2 graph:

* ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) - joint names and
  positions, every control step.
* ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``) - one
  message per attached camera frame.

``rclpy`` and the ROS 2 message packages are optional, system-provided
dependencies (they are not on PyPI); they are imported lazily, so importing this
module - and running hardware with ``ros2_bridge=False`` - never requires ROS 2.
"""

from __future__ import annotations

from strands_robots.ros_telemetry import RosTelemetryBridge


class HardwareRosBridge(RosTelemetryBridge):
    """Telemetry bridge for a real robot (node name ``strands_hardware``).

    Identical wire behavior to its simulation sibling
    :class:`~strands_robots.simulation.ros_bridge.SimRosBridge`; only the
    default rclpy node name differs so the two are distinguishable on the graph.
    See :class:`~strands_robots.ros_telemetry.RosTelemetryBridge` for the full
    publish API and constructor arguments.
    """

    default_node_name = "strands_hardware"
