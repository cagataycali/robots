#!/usr/bin/env python3
"""Expose a real robot on a ROS 2 domain (Robot ros2_bridge=True).

Goal: Show that a physical arm can publish its live observation on ROS 2 so
external ROS 2 nodes (rviz, nav2, or the agent's own use_ros calls) can
subscribe to the hardware. This is the symmetric counterpart of
examples/ros2/sim_bridge_demo.py - the sim and hardware bridges are thin
subclasses of the same RosTelemetryBridge, so a real arm and its digital twin
publish identical topics. With ``ros2_bridge=True`` the robot advertises, per
robot:

  /<robot>/joint_states            sensor_msgs/msg/JointState  (every control step)
  /<robot>/<camera>/image_raw      sensor_msgs/msg/Image  (rgb8, per camera)

Dependencies:
  pip install "strands-robots[ros2]"
  rclpy must be importable - it ships with a system ROS 2 install (apt /
  RoboStack) or the official docker images, not PyPI. Run this in an
  environment where `python3 -c "import rclpy"` works (e.g. inside a
  `ros:jazzy` container with strands-robots installed).

A physical SO-101 must be connected; pass its serial port. With no arm attached
this raises at connect() - the bridge itself is hardware-agnostic.

Verify from another shell on the same ROS 2 domain:
  ros2 topic list | grep so101
  ros2 topic echo /so101/joint_states
"""

import time

from strands_robots import Robot

# ros2_bridge=True spins up an internal rclpy node; ros2_domain picks the domain.
# Opt-in by design: ros2_bridge=False (the default) never touches ROS 2.
arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_domain=0,
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}},
)

print("publishing /so101/joint_states (+ /so101/wrist/image_raw) on domain 0")

try:
    # Publish the live observation on demand, ~10 Hz. (Inside a running task the
    # control loop publishes automatically after each observation.)
    while True:
        arm.publish_ros_observation()
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    arm.cleanup()  # tears down the ROS 2 node cleanly
