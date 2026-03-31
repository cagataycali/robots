#!/usr/bin/env python3
"""Record simulation rollouts as LeRobot v3 datasets.

Runs a mock policy in MuJoCo, captures joint states + video, and saves
everything as a LeRobot-compatible dataset (parquet + AV1 video).

Requirements:
    pip install strands-robots[sim]

Usage:
    python examples/03_sim_recording.py
"""

from strands_robots import Robot

sim = Robot("so100")

# Start recording — creates LeRobot v3 dataset structure
sim.start_recording(
    repo_id="local/so100_demo",
    task="reach target",
    fps=50,
    root="/tmp/so100_dataset",
)

# Run a mock policy (random actions) for 2 seconds
result = sim.run_policy(
    robot_name="so100",
    policy_provider="mock",
    instruction="reach target",
    duration=2.0,
    fast_mode=True,
    record_video="/tmp/so100_rollout.mp4",
    video_fps=30,
)
print(result["content"][0]["text"])

# Finalize the episode
stop = sim.stop_recording()
print(stop["content"][0]["text"])

sim.destroy()
print("✅ Dataset saved to /tmp/so100_dataset/")
