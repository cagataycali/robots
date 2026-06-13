#!/usr/bin/env python3
"""Record a demonstration as a LeRobotDataset — from sim, push to HF Hub.

Goal: Show the recording lifecycle (start -> run policy -> stop) produces
a training-ready LeRobotDataset with zero manual feature wrangling.

Dependencies: pip install "strands-robots[sim-mujoco,lerobot]"
Expected output: Dataset saved locally with 1 episode, 100 frames.
Runtime: ~3 seconds.
"""

from strands_robots import MockPolicy, Robot

sim = Robot("so100", mesh=False)
sim.create_world()
sim.add_robot(name="arm", data_config="so100")
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])

# Start recording — features are auto-inferred from the robot.
sim.start_recording(repo_id="local/my_demo", fps=30, task="pick up the red cube")

# Run the policy — each step is automatically captured.
sim.run_policy(
    robot_name="arm",
    policy_object=MockPolicy(),
    instruction="pick up the red cube",
    n_steps=100,
)

# Finalize — writes parquet + video, ready for lerobot training scripts.
result = sim.stop_recording()
print(f"Recording: {result['status']}")

# To push to HF Hub instead: use repo_id="your_user/dataset_name"
# and call sim.stop_recording(push_to_hub=True) with HF_TOKEN set.
