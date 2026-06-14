#!/usr/bin/env python3
"""Attach an LLM agent to a robot — control it with natural language.

Goal: Show that a Strands Agent with Robot() as a tool lets you describe
tasks in English. The agent composes sim scenes, runs policies, and records
datasets — all from one prompt.

Dependencies:
  pip install "strands-robots[sim-mujoco]" strands-agents
  AWS credentials for Bedrock (or any strands-agents model provider).

Expected output: Agent creates a scene and runs a policy from a text prompt.
Runtime: ~10 seconds (depends on LLM latency).
"""

from strands import Agent

from strands_robots import Robot

# Robot() is a Strands AgentTool — pass it directly to the Agent.
sim = Robot("so100", mesh=False)

agent = Agent(tools=[sim])

# One prompt drives the full workflow: scene setup + policy execution.
result = agent(
    "Create a world with the so100 robot, add a small red cube at [0.2, 0, 0.05], "
    "add a front camera looking at it, then run the Mock policy for 30 steps "
    "with instruction 'pick up the red cube'."
)

print(f"Agent completed: {result}")
