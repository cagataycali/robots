#!/usr/bin/env python3
"""Run a HuggingFace ACT policy in MuJoCo and export a LeRobot dataset.

Downloads a pretrained ACT policy, runs it in simulation, records multi-camera
video + joint data as a LeRobot v3 dataset. The full pipeline in ~20 lines.

Requirements:
    pip install strands-robots[sim] lerobot torch

Usage:
    python examples/act_policy_simulation.py
"""

from strands_robots import Robot

# 1. Create simulated Aloha bimanual robot (14 actuators, 6 cameras)
sim = Robot("aloha")

# 2. Start recording a LeRobot dataset (parquet + AV1 video)
sim.start_recording(
    repo_id="local/act_aloha_sim_demo",
    task="transfer cube",
    fps=50,
    root="/tmp/act_aloha_dataset",
)

# 3. Run a pretrained ACT policy from HuggingFace (51M params)
# NOTE: This downloads model weights (~200MB) on first run.
# For a lightweight test without downloading, use policy_provider="mock":
#   sim.run_policy(robot_name="aloha", policy_provider="mock", duration=2.0)
result = sim.run_policy(
    robot_name="aloha",
    policy_provider="lerobot_local",
    pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
    instruction="transfer cube",
    duration=2.0,  # seconds of sim time
    fast_mode=True,  # no wall-clock sleep between steps
    record_video="/tmp/act_aloha_rollout.mp4",
)
print(result["content"][0]["text"])

# 4. Save the episode to disk
stop = sim.stop_recording()
print(stop["content"][0]["text"])

sim.destroy()
