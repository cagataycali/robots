#!/usr/bin/env python3
"""Control real robot hardware with the same factory API.

Robot(mode="real") returns a HardwareRobot backed by LeRobot. The same
Agent workflow works — just swap the mode.

Requirements:
    pip install strands-agents strands-robots[lerobot]
    # Hardware: SO-100/SO-101 arm connected via USB (Feetech servos)

Usage:
    # Auto-detect (switches to real if USB servo controller found)
    STRANDS_ROBOT_MODE=real python examples/04_real_hardware.py

    # Or set mode explicitly in code (see below)
"""

from strands import Agent

from strands_robots import Robot

# Explicit real mode with camera config
robot = Robot(
    "so100",
    mode="real",
    cameras={
        "wrist": {
            "type": "opencv",
            "index_or_path": "/dev/video0",
            "fps": 15,
            "fourcc": "MJPG",
        },
    },
)

# Same Agent interface as simulation
agent = Agent(tools=[robot])
agent("Connect to the robot, read the current joint positions, and report status")
