#!/usr/bin/env python3
"""Record simulation rollouts as LeRobot v3 datasets.

Runs a mock policy in MuJoCo, captures joint states + video, and saves
everything as a LeRobot-compatible dataset (parquet + AV1 video).

Note:
    ``start_recording`` produces a LeRobotDataset (parquet + per-camera
    MP4), so you need the ``lerobot`` extra in addition to ``sim-mujoco``.
    For plain MP4 only (no dataset schema), use ``start_cameras_recording``.

Requirements:
    pip install "strands-robots[sim-mujoco,lerobot]"

Usage:
    python examples/03_sim_recording.py
"""

from strands_robots import Robot

sim = Robot("so100")

# Start recording - creates LeRobot v3 dataset structure
sim.start_recording(
    repo_id="local/so100_demo",
    task="reach target",
    fps=50,
    root="/tmp/so100_dataset",
)

# Run a mock policy (random actions) for 2 seconds.
# Video kwargs go inside the ``video`` dict, NOT as top-level args.
result = sim.run_policy(
    robot_name="so100",
    policy_provider="mock",
    instruction="reach target",
    duration=2.0,
    fast_mode=True,
    video={"path": "/tmp/so100_rollout.mp4", "fps": 30},
)
print(result["content"][0]["text"])

# Finalize the episode
stop = sim.stop_recording()
print(stop["content"][0]["text"])

sim.destroy()
print("Dataset saved to /tmp/so100_dataset/")
