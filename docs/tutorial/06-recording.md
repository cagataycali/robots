---
description: start_recording, run a policy, stop_recording — produce a LeRobot v3 dataset on disk.
---

# 6 — Recording data

Record a session of robot motion (joints + cameras + actions) into a LeRobot v3 dataset.
The dataset is parquet + MP4 video and is ready for the LeRobot training loop.

## TL;DR

```python
from strands_robots import Robot

sim = Robot("so100")
sim.add_camera(name="wrist", attach_to="so100", pos=[0.05, 0, 0.1], fovy=60)
sim.add_object(name="cube", type="box", size=[0.025]*3, pos=[0.3, 0, 0.025],
               rgba=[1, 0, 0, 1])

# Start recording — every step is captured
sim.start_recording(repo_id="user/my_dataset", task="pick up the red cube", fps=30)

sim.run_policy(instruction="pick up the cube", policy_provider="mock", duration=10.0)

sim.stop_recording()
# /tmp/my_dataset/   (LeRobot v3 layout)
```

## Setup

Recording requires `lerobot`. Install with the lerobot extra:

```bash
pip install "strands-robots[lerobot]"
```

Without `lerobot` installed, `start_recording` returns an error explaining the missing
dependency.

## What gets recorded

For every control step the recorder appends one frame containing:

- **Observation**: every camera frame as MP4-encoded video, plus joint state.
- **Action**: whatever the policy returned (joint positions, deltas, etc.).
- **Task**: the natural-language instruction, repeated per frame for indexing.
- **Timestamp**: monotonic time relative to episode start.

The schema is auto-derived from the simulation's `observation_features` and
`action_features` — you don't have to declare anything by hand.

## On-disk layout (LeRobot v3)

```
/tmp/my_dataset/
├── meta/
│   ├── info.json              — dataset metadata (fps, total_frames, total_episodes)
│   ├── tasks.parquet          — task descriptions table
│   ├── episodes.parquet       — episode boundaries
│   └── stats.parquet          — per-feature stats (auto-computed)
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  — observation.state + action columns
└── videos/
    └── chunk-000/
        └── observation.images.{cam}/
            └── episode_*.mp4  — one MP4 per camera per episode
```

This is the LeRobot v3 standard. Dump it into `LeRobotDataset.create(...)` as-is and
train. The chunk size and video codec are configurable through the recording action.

## The action surface

`Simulation` exposes three recording actions:

- `start_recording(repo_id, task, fps=30, ...)` — begin a recording session.
- `stop_recording()` — close the open episode, write meta files, return paths.
- `get_recording_status()` — current episode id, frame count, target fps, output dir.

Behind the scenes these wrap `strands_robots.dataset_recorder.DatasetRecorder`. You can
also instantiate the recorder directly for non-sim use cases — see
[Recording reference](../recording.md).

## Multi-episode recording

```python
for episode_idx in range(10):
    sim.reset()
    sim.randomize(colors=True, lighting=True)

    sim.start_recording(
        repo_id="user/my_dataset",
        task=f"episode {episode_idx}: pick up the cube",
        fps=30,
        episode_id=episode_idx,
    )
    sim.run_policy(instruction="pick up the cube", policy_provider="mock", duration=10.0)
    sim.stop_recording()

# 10 episodes appended to /tmp/my_dataset/data/chunk-000/
```

You can also use `eval_policy(num_episodes=N, record=True, ...)` which manages the
episode loop for you.

## Pushing to the Hub

After recording locally, push:

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(repo_id="user/my_dataset",
                         root="/tmp/my_dataset")
dataset.push_to_hub()
```

You'll need `huggingface-cli login` first. Once on the Hub, anyone can:

```python
LeRobotDataset(repo_id="user/my_dataset", local_files_only=False)
```

## Replay

Replay an episode back into the simulation to verify it captured correctly:

```python
sim.replay_episode(repo_id="user/my_dataset", episode_id=0)
```

`replay_episode` reads the recorded actions from disk and re-runs them in the current
sim. Useful for debugging policy traces.

## Recap

- `start_recording` → `run_policy` → `stop_recording`.
- Output is LeRobot v3 (parquet + MP4) — train without conversion.
- Schema is auto-derived from the sim's observation/action features.
- `eval_policy` can drive the multi-episode loop for you.
- `replay_episode` plays a recording back into sim.

## See also

- [Recording reference](../recording.md) — the underlying `DatasetRecorder` class and
  every parameter `start_recording` accepts.
- [Tutorial 7 — Training](07-training.md) — what to do with the dataset you just made.
- [LeRobot dataset format](https://huggingface.co/docs/lerobot) — the upstream spec.
