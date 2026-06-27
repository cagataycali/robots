"""Publish simulation telemetry on a ROS 2 domain.

When a :class:`~strands_robots.simulation.base.SimEngine` is constructed with
``ros2_bridge=True``, it owns a :class:`SimRosBridge` that advertises the sim's
live state on a ROS 2 domain so external ROS 2 nodes can
``ros2 topic echo /<robot>/joint_states`` against the simulation - and the
agent's own ``use_ros`` calls reach the same graph, closing the loop end to end.

:class:`SimRosBridge` is the simulation half of a symmetric pair: the hardware
half is :class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`. Both are
thin subclasses of :class:`strands_robots.ros_telemetry.RosTelemetryBridge` and
publish the identical per-robot topics, so a simulated robot and the real arm it
mirrors are indistinguishable on the ROS 2 graph:

* ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) - joint names and
  positions, every sim step.
* ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``) - one
  message per attached camera that rendered a frame.

``rclpy`` and the ROS 2 message packages are optional, system-provided
dependencies (they are not on PyPI); they are imported lazily, so importing this
module - and running the simulation with ``ros2_bridge=False`` - never requires
ROS 2.
"""

from __future__ import annotations

from strands_robots.ros_telemetry import RosTelemetryBridge


class SimRosBridge(RosTelemetryBridge):
    """Telemetry bridge for a running simulation (node name ``strands_sim``).

    Identical wire behavior to its hardware sibling
    :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge`; only the
    default rclpy node name differs so the two are distinguishable on the graph.
    See :class:`~strands_robots.ros_telemetry.RosTelemetryBridge` for the full
    publish API and constructor arguments.
    """

    default_node_name = "strands_sim"
