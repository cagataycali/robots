#!/usr/bin/env python3
"""Expose a MuJoCo simulation on a ROS 2 domain (SimEngine ros2_bridge=True).

Goal: Show that the simulator can publish its live state on ROS 2 so external
ROS 2 nodes (or the agent's own use_ros calls) can subscribe to it. With
``ros2_bridge=True`` the sim advertises, per robot:

  /<robot>/joint_states            sensor_msgs/msg/JointState  (every step)
  /<robot>/<camera>/image_raw      sensor_msgs/msg/Image  (rgb8, per camera)

Dependencies:
  pip install "strands-robots[sim-mujoco,ros2]"
  rclpy must be importable - it ships with a system ROS 2 install (apt /
  RoboStack) or the official docker images, not PyPI. Run this script in an
  environment where `python3 -c "import rclpy"` works (e.g. inside a
  `ros:jazzy` container with mujoco + strands-robots installed).

Verify from another shell on the same ROS 2 domain:
  ros2 topic list | grep so101
  ros2 topic echo /so101/joint_states

Runtime: runs until interrupted.
"""

import math
import time

from strands_robots.simulation import Simulation

# ros2_bridge=True spins up an internal rclpy node; ros2_domain picks the domain.
sim = Simulation(ros2_bridge=True, ros2_domain=0)
sim.create_world()
sim.add_robot("so101")

joints = sim.robot_joint_names("so101")
print(f"publishing /so101/joint_states for joints: {joints}")

try:
    i = 0
    while True:
        target = 0.6 * math.sin(i / 40.0)
        sim.send_action({j: target for j in joints}, robot_name="so101")
        sim.step(5)  # each step publishes joint_states (+ camera image_raw)
        time.sleep(0.05)
        i += 1
except KeyboardInterrupt:
    pass
finally:
    sim.destroy()  # tears down the ROS 2 node cleanly
