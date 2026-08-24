#!/usr/bin/env python3
"""Drive an AWS DeepRacer with natural language - Ackermann bridge demo.

Goal: Show AckermannRosRobot presenting a steering-geometry ROS 2 car as
strands agent tools. The bridge converts (linear, angular) commands through a
bicycle model to the DeepRacer's normalized servo interface, runs the car's
manual-mode service handshake automatically before the first command, and
always trails sustained commands with a zero so the car cannot be left
driving.

Dependencies:
  pip install strands-robots strands-agents
  A sourced ROS 2 environment (rclpy) on the same network/domain as the car,
  with the DeepRacer core stack running (ctrl_pkg, webserver_pkg, rplidar).

SAFETY: put the car on blocks or a clear track before running. The servo topic
and both mode services are gated command surfaces, so this agent prompts for
operator approval before each command it sends. For an unattended run,
pre-approve exactly those three surfaces instead (bare names cover the
namespaced DeepRacer spellings):

    export STRANDS_ROS2_COMMAND_ALLOW=/manual_drive,/vehicle_state,/enable_state

Expected output: the agent drives a short pattern and summarizes the lidar.
Runtime: ~20 seconds (depends on LLM latency).
"""

from strands import Agent

from strands_robots.mesh import AckermannRosRobot

# Stock DeepRacer wiring: servo topic, lidar, and the two-step manual-mode
# handshake are preconfigured; override any keyword for a modified car.
car = AckermannRosRobot.from_deepracer(node_name="deepracer")

agent = Agent(tools=car.tools)

result = agent(
    "Drive forward at 0.3 m/s for two seconds, then arc gently left for two "
    "seconds, stop, and read the lidar to tell me the nearest obstacle."
)

print(f"Agent completed: {result}")
