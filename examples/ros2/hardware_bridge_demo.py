#!/usr/bin/env python3
"""Expose a real robot on a ROS 2 domain - full duplex (Robot ros2_bridge=True).

Goal: Show that a physical arm becomes a first-class ROS 2 participant in BOTH
directions when constructed with ``ros2_bridge=True``:

  * publish (outbound) - the live observation is advertised so external ROS 2
    nodes (rviz, nav2, or the agent's own use_ros calls) can subscribe to the
    hardware, per robot:

      /<robot>/joint_states            sensor_msgs/msg/JointState  (every control step)
      /<robot>/<camera>/image_raw      sensor_msgs/msg/Image  (rgb8, per camera)

  * subscribe (inbound) - the bridge listens on /<robot>/joint_command and
    forwards each JointState into Robot.send_action, so an external ROS 2 stack
    (a teleop node, MoveIt, a trajectory replayer) can DRIVE the real arm:

      /<robot>/joint_command           sensor_msgs/msg/JointState  (name+position -> send_action)

This is the symmetric counterpart of examples/ros2/sim_bridge_demo.py - the sim
and hardware telemetry bridges are thin subclasses of the same
RosTelemetryBridge, so a real arm and its digital twin publish identical topics.
The inbound command path lives only on the hardware bridge: a simulation is
driven by its physics engine; the real arm is the thing an external controller
can physically move.

Dependencies:
  pip install "strands-robots[ros2]"
  rclpy must be importable - it ships with a system ROS 2 install (apt /
  RoboStack) or the official docker images, not PyPI. Run this in an
  environment where `python3 -c "import rclpy"` works (e.g. inside a
  `ros:jazzy` container with strands-robots installed).

A physical SO-101 must be connected; pass its serial port. With no arm attached
this raises at connect() - the bridge itself is hardware-agnostic.

Verify telemetry from another shell on the same ROS 2 domain:
  ros2 topic list | grep so101
  ros2 topic echo /so101/joint_states

Drive the arm from another shell (inbound command path):
  ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
    '{name: ["shoulder_pan.pos", "elbow.pos"], position: [0.1, -0.2]}'
"""

import time

from strands_robots import Robot

# ros2_bridge=True spins up an internal rclpy node; ros2_domain picks the domain.
# ros2_commands=True (the default) also subscribes to /so101/joint_command and
# drives the arm - full duplex. Set ros2_commands=False for a read-only bridge.
# Opt-in by design: ros2_bridge=False (the default) never touches ROS 2.
arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_domain=0,
    ros2_commands=True,  # subscribe to joint_command and drive the arm
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}},
)

print("publishing /so101/joint_states (+ /so101/wrist/image_raw) on domain 0")
print("listening on /so101/joint_command to drive the arm (full duplex)")

try:
    # Publish the live observation on demand, ~10 Hz. (Inside a running task the
    # control loop publishes automatically after each observation.) Meanwhile a
    # background spin thread services inbound joint_command messages and drives
    # the arm via send_action - both directions run concurrently.
    while True:
        arm.publish_ros_observation()
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    arm.cleanup()  # stops the command thread and tears down the ROS 2 node cleanly
