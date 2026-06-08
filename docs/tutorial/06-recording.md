---
description: start_recording, run a policy, stop_recording — produce a LeRobot v3 dataset on disk.
---

# 6 — Recording data

```python
from strands_robots import Robot

sim = Robot("so100")
sim.add_camera(name="wrist", position=[0.05, 0, 0.1], target=[0.3, 0, 0.1], fov=60)
sim.add_object(name="cube", shape="box", size=[0.025]*3,
               position=[0.3, 0, 0.025], color=[1, 0, 0, 1])

sim.start_recording(repo_id="user/my_dataset", task="pick up the red cube", fps=30)

sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)

sim.stop_recording()   # LeRobot v3 parquet + MP4 written to disk
```

```bash
pip install "strands-robots[lerobot]"   # required for start_recording
```

## Multi-episode

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube",
                    fps=30, overwrite=True)

for _ in range(10):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_lighting=True)
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", duration=10.0)
    # one episode saved automatically after each run_policy

sim.stop_recording()   # 10 episodes, metadata finalized
```

## Output layout (LeRobot v3)

```
my_dataset/
├── meta/           info.json  tasks.parquet  episodes.parquet  stats.parquet
├── data/chunk-000/ episode_*.parquet   (observation.state + action columns)
└── videos/chunk-000/observation.images.{cam}/episode_*.mp4
```

Schema is auto-derived from the sim's observation/action features — no manual declaration.

## Plain MP4 alternative (no lerobot needed)

```python
# [sim-mujoco] only
sim.start_cameras_recording(output_dir="my_recording", fps=30)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_cameras_recording()   # my_recording/{camera_name}.mp4
```

## Replay & push

```python
# Verify a recording by replaying it into the sim
sim.replay_episode(repo_id="user/my_dataset", robot_name="so100", episode=0)

# Push to Hugging Face Hub (huggingface-cli login first)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(repo_id="user/my_dataset", root="my_dataset").push_to_hub()
```

## See also

- [Recording reference](../recording.md) — `DatasetRecorder` class, every parameter.
- [Tutorial 7 — Training](07-training.md) — what to do with the dataset.
- [LeRobot dataset format](https://huggingface.co/docs/lerobot) — upstream spec.
