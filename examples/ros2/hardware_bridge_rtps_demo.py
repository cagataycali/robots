#!/usr/bin/env python3
"""Expose a real robot on ROS 2 over pure RTPS - no rclpy, no sourced distro.

This is the rclpy-free twin of examples/ros2/hardware_bridge_demo.py. Same
full-duplex behavior - publish /<robot>/joint_states (+ camera image_raw),
subscribe /<robot>/joint_command -> send_action - but the transport is pure
cyclonedds (a single pip wheel) instead of rclpy, so it runs with NO sourced
ROS 2 distro:

    pip install "strands-robots[ros2]"   # pulls cyclonedds; nothing else needed

The two transports emit byte-identical topics, so a real ROS 2 node, rviz, or
the stock `ros2` CLI cannot tell this apart from the rclpy bridge or a real
hardware node:

    ros2 topic echo /so101/joint_states
    ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
      '{name: ["shoulder_pan.pos"], position: [0.1]}'

Type coverage is bounded by the RTPS IDL bundle (strands_robots.rtps.idl):
joint_states + image_raw are in; anything outside the bundle needs the rclpy
backend (ros2_transport="rclpy", the default).

A physical SO-101 must be connected; with no arm attached this raises at
connect() - the bridge itself is hardware-agnostic.
"""

import time

from strands_robots import Robot

# ros2_transport="rtps" selects the cyclonedds backend (no rclpy). Everything
# else is identical to the rclpy hardware bridge demo.
arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_transport="rtps",
    ros2_domain=0,
    ros2_commands=True,  # subscribe to joint_command and drive the arm
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}},
)

print("publishing /so101/joint_states (+ /so101/wrist/image_raw) over RTPS, no rclpy")
print("listening on /so101/joint_command to drive the arm (full duplex)")

try:
    while True:
        arm.publish_ros_observation()
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    arm.cleanup()  # stops the poll thread and drops the cyclonedds participant
