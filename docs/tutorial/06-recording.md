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
sim.add_camera(name="wrist", position=[0.05, 0, 0.1], target=[0.3, 0, 0.1], fov=60)
sim.add_object(name="cube", shape="box", size=[0.025]*3,
               position=[0.3, 0, 0.025], color=[1, 0, 0, 1])

# Requires [lerobot] extra — see Setup below
sim.start_recording(repo_id="user/my_dataset", task="pick up the red cube", fps=30)

sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)

sim.stop_recording()
# LeRobot v3 dataset written to the repo directory
```

## Setup

`start_recording` requires the `[lerobot]` extra (LeRobot v3 = parquet + MP4).
Install it with:

```bash
pip install "strands-robots[lerobot]"
```

Without `lerobot` installed, `start_recording` returns a friendly error pointing you
to `start_cameras_recording` (the plain-MP4 alternative — see below).

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
my_dataset/
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

- `start_recording(repo_id, task, fps=30, overwrite=False, ...)` — begin a recording session.
- `stop_recording()` — close the open episode, write meta files, return paths.
- `get_recording_status()` — current episode id, frame count, target fps, output dir.

`start_recording` and `stop_recording` manage a `DatasetRecorder` internally.
Per-episode, you call `reset()` and `run_policy(robot_name=...)`; the recorder saves
one episode automatically after each `run_policy` call completes, and `stop_recording`
finalises all metadata.

Behind the scenes these wrap `strands_robots.dataset_recorder.DatasetRecorder`. You can
also instantiate the recorder directly for non-sim use cases — see
[Recording reference](../recording.md).

## Multi-episode recording

```python
sim.start_recording(
    repo_id="user/my_dataset",
    task="pick up the cube",
    fps=30,
    overwrite=True,   # create fresh dataset on first run
)

for _ in range(10):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_lighting=True)
    sim.run_policy(
        robot_name="so100",
        instruction="pick up the cube",
        policy_provider="mock",
        duration=10.0,
    )
    # The internal DatasetRecorder saves one episode after each run_policy call.

sim.stop_recording()
# 10 episodes appended to the dataset
```

`start_recording`/`stop_recording` manage the `DatasetRecorder` internally.
Call `start_recording` once before the loop; the recorder accumulates episodes
until `stop_recording` finalises the metadata files.

## Plain MP4 alternative (no lerobot required)

If you only need raw video files and do not need the LeRobot parquet format,
use `start_cameras_recording`. This works under the `[sim-mujoco]` extra alone —
no `lerobot` installation required:

```bash
pip install "strands-robots[sim-mujoco]"
```

```python
# Plain MP4 per camera — [sim-mujoco] only, no lerobot needed
sim.start_cameras_recording(output_dir="my_recording", fps=30)

sim.run_policy(
    robot_name="so100",
    instruction="pick up the cube",
    policy_provider="mock",
    duration=10.0,
)

sim.stop_cameras_recording()
# my_recording/{camera_name}.mp4 written for each camera
```

Use `get_cameras_recording_status()` to inspect the current state.

## Pushing to the Hub

After recording locally, push the LeRobot dataset to the Hugging Face Hub:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="user/my_dataset",
                         root="my_dataset")
dataset.push_to_hub()
```

You'll need `huggingface-cli login` first. Once on the Hub, anyone can:

```python
LeRobotDataset(repo_id="user/my_dataset", local_files_only=False)
```

## Replay

Replay an episode back into the simulation to verify it captured correctly:

```python
sim.replay_episode(repo_id="user/my_dataset", robot_name="so100", episode=0)
```

`replay_episode` reads the recorded actions from disk and re-runs them in the current
sim. Useful for debugging policy traces.

## Recap

- `start_recording` requires `[lerobot]`; `start_cameras_recording` is the plain-MP4
  alternative and works under `[sim-mujoco]` alone.
- `start_recording` → loop `reset()` + `run_policy(robot_name=...)` → `stop_recording`.
- Output is LeRobot v3 (parquet + MP4) — train without conversion.
- Schema is auto-derived from the sim's observation/action features.
- `replay_episode(repo_id=..., robot_name=..., episode=0)` plays a recording back into sim.

## See also

- [Recording reference](../recording.md) — the underlying `DatasetRecorder` class and
  every parameter `start_recording` accepts.
- [Tutorial 7 — Training](07-training.md) — what to do with the dataset you just made.
- [LeRobot dataset format](https://huggingface.co/docs/lerobot) — the upstream spec.
