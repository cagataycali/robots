---
description: DatasetRecorder — LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
---

# Recording & datasets

`strands_robots.dataset_recorder.DatasetRecorder` writes LeRobot v3 datasets — the
format `lerobot` consumes for training. It's used by both `Simulation` and
`HardwareRobot` through the `start_recording` / `stop_recording` actions.

## TL;DR

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
sim.run_policy(instruction="pick up the cube", policy_provider="mock", duration=10.0)
sim.stop_recording()

# /tmp/my_dataset is a LeRobot v3 dataset
```

## Direct API

For non-sim use cases (recording from a custom control loop), instantiate the recorder
directly:

```python
from strands_robots.dataset_recorder import DatasetRecorder

recorder = DatasetRecorder.create(
    repo_id="user/my_dataset",
    fps=30,
    robot_features=robot.observation_features,
    action_features=robot.action_features,
    task="pick up the red cube",
)

# In your control loop:
recorder.add_frame(observation, action, task="pick up the red cube")

# End of episode:
recorder.save_episode()

# Optional:
recorder.push_to_hub()
```

## On-disk layout (LeRobot v3)

```
{root}/
├── meta/
│   ├── info.json              — dataset metadata (fps, total_frames, total_episodes)
│   ├── tasks.parquet          — task descriptions
│   ├── episodes.parquet       — episode boundaries
│   └── stats.parquet          — per-feature statistics (auto-computed)
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  — observation.state + action columns
└── videos/
    └── chunk-000/
        └── observation.images.{cam}/
            └── episode_*.mp4
```

The schema is auto-derived from the simulation's or hardware robot's
`observation_features` and `action_features`. You don't declare it by hand.

## When LeRobot is missing

`dataset_recorder.py` checks for `lerobot` lazily and raises a clear error if it's
missing:

```bash
pip install "strands-robots[lerobot]"
```

The check is wrapped (`has_lerobot_dataset()`) and cached so subsequent calls are
cheap. On Jetson / JetPack systems the check also handles the
`numpy ABI mismatch` failure mode by returning False without raising — see the
[installation notes](getting-started/installation.md).

## Multi-camera recording

When a `Simulation` has multiple cameras attached (or a `HardwareRobot` has a
multi-cam config), every camera goes into its own MP4 stream:

```
videos/chunk-000/
├── observation.images.wrist/episode_000000.mp4
├── observation.images.front/episode_000000.mp4
└── observation.images.top/episode_000000.mp4
```

The frame rate is shared (`fps=30` across all cameras). Resolution is per-camera —
whatever you set in `add_camera(...)` (sim) or in the `cameras=` kwarg (real).

## Reading the dataset back

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds))                      # number of frames
print(ds[0].keys())                 # observation/action features
print(ds.meta.fps, ds.meta.episodes)
```

This is exactly the input format LeRobot's training scripts consume.

## Pushing to the Hub

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
ds.push_to_hub()                    # requires huggingface-cli login first
```

After that, anyone can `LeRobotDataset(repo_id="user/my_dataset")` and pull it.

## See also

- [Tutorial 6 — Recording](tutorial/06-recording.md) — guided walkthrough.
- [Tutorial 7 — Training](tutorial/07-training.md) — what to do with the recorded
  data.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) — upstream spec.
- [Simulation overview](simulation/overview.md) — `start_recording` / `stop_recording`
  parameters.
