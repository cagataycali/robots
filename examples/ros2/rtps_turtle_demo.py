#!/usr/bin/env python3
"""Drive a ROS 2 turtle over PURE RTPS - no rclpy, no sourced ROS 2 distro.

This is the "act as a robot" demo: a bare cyclonedds participant publishes
geometry_msgs/Twist on /turtle1/cmd_vel, and a real ROS 2 turtlesim moves. The
only dependency is the pip-installable cyclonedds wheel - this script runs on
macOS / Jetson / CI with NO ROS 2 installed in the Python environment.

Dependencies:
  pip install "strands-robots[ros2]"   # cyclonedds, a self-contained wheel

A ROS 2 turtle on the same DDS domain (e.g. host networking):
  docker run -d --name turtle --net host ros:jazzy bash -lc \\
    "apt-get update && apt-get install -y ros-jazzy-turtlesim && \\
     source /opt/ros/jazzy/setup.bash && \\
     QT_QPA_PLATFORM=offscreen ros2 run turtlesim turtlesim_node"

Expected: the turtle drives forward and to the left for ~1.5 seconds.
Runtime: ~2 seconds.
"""

from strands_robots.mesh import RtpsRobot

# 1. Wrap the turtle as a pure-RTPS robot. No ROS 2 needed in this interpreter.
turtle = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")

# 2. Appear on the ROS 2 graph as a cmd_vel publisher, then drive.
turtle.advertise()
turtle.drive(linear=2.0, angular=1.5, duration=1.5)  # publish Twist over RTPS
turtle.stop()
print("done - the turtle should have moved (check the turtlesim window)")

# 3. Or hand the robot to an agent. Its methods become named tools
#    (drive_turtlesim, stop_turtlesim). Uncomment to run with a model provider:
#
# from strands import Agent
# agent = Agent(tools=turtle.tools)
# agent("drive forward for two seconds while turning left, then stop")
