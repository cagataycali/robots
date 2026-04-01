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

# Factory creates a MuJoCo sim (auto-downloads assets on first run)
robot = Robot("so100")

# The sim IS the tool — pass it directly to Agent
agent = Agent(tools=[robot])

# Natural language → simulation actions
result = agent("Get the simulation state, then run a mock policy for 1 second in fast mode")
print(result)

robot.destroy()
