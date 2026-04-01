#!/usr/bin/env python3
"""Run NVIDIA GR00T policy on real hardware.

Starts a GR00T inference server, connects to a real SO-101 arm with
dual cameras, and runs the policy through an Agent.

Requirements:
    pip install strands-agents strands-robots[all]
    # Hardware: SO-101 arm + 2 USB cameras
    # Model: Download from HuggingFace (e.g., cagataydev/gr00t-wave)

Usage:
    python examples/05_real_groot_policy.py
"""

from strands import Agent

from strands_robots import Robot, gr00t_inference, lerobot_camera, pose_tool

# Real robot with dual cameras
robot = Robot(
    "so101",
    mode="real",
    cameras={
        "wrist": {
            "type": "opencv",
            "index_or_path": "/dev/video0",
            "fps": 15,
            "fourcc": "MJPG",
        },
        "front": {
            "type": "opencv",
            "index_or_path": "/dev/video2",
            "fps": 15,
            "fourcc": "MJPG",
        },
    },
)

# Build agent with robot + inference tools
agent = Agent(
    tools=[robot, gr00t_inference, lerobot_camera, pose_tool],
)

# Start GR00T inference server
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
    port=5555,
    data_config="so100_dualcam",
    embodiment_tag="new_embodiment",
)

# Interactive control loop
print("GR00T policy running. Type instructions or 'quit' to exit.")
while True:
    query = input("\n# ")
    if query.lower() in ("quit", "exit", "q"):
        break
    agent(query)

agent.tool.gr00t_inference(action="stop", port=5555)
