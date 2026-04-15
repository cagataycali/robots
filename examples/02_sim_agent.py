#!/usr/bin/env python3
"""The 5-Line Promise: natural language robot control.

Robot() returns an AgentTool with 35+ simulation actions. Hand it to a
Strands Agent and control the robot through conversation.

Requirements:
    pip install strands-agents strands-robots[sim]

Usage:
    python examples/02_sim_agent.py
"""

from strands import Agent

from strands_robots import Robot

# Robot("so100") auto-detects mode="sim", picks the "mujoco" backend,
# constructs a Simulation instance, calls add_robot() to load the SO-100
# model (auto-downloading URDF/meshes on first run), and returns that
# Simulation as an AgentTool. You get full access to all Simulation
# actions — step(), render(), run_policy(), get_observation(), etc.
robot = Robot("so100")

# The sim IS the tool — pass it directly to Agent
agent = Agent(tools=[robot])

# Natural language → simulation actions
result = agent("Get the simulation state, then run a mock policy for 1 second in fast mode")
print(result)

robot.destroy()
