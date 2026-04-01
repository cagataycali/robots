#!/usr/bin/env python3
"""Quickstart: 5 lines from import to simulation.

The Robot() factory auto-detects mode (defaults to sim when no hardware
is connected) and returns a ready-to-use MuJoCo simulation.

Requirements:
    pip install strands-robots[sim]

Usage:
    python examples/01_sim_quickstart.py
"""

from strands_robots import Robot

# Create a simulated SO-100 arm — assets auto-download from MuJoCo Menagerie
sim = Robot("so100")

# Inspect the world
state = sim.get_state()
print(state["content"][0]["text"])

# Step physics and render a frame
sim.step(n_steps=100)
frame = sim.render(width=640, height=480)
print(f"Rendered frame: {frame['content'][0]['text']}")

sim.destroy()
print("Done — simulation complete")
