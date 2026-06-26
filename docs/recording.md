---
description: DatasetRecorder - LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
---

# Recording & datasets

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_recording()
# LeRobot v3 dataset written to ~/.strands_robots/datasets/
```

`start_recording` requires `[lerobot]`. Without it, use `start_cameras_recording` for plain MP4.

## Multi-episode recording

A recording session is one dataset; `run_policy` alone does **not** delimit
episodes. To collect N episodes in one session, call `save_episode()` after each
rollout to flush it as its own episode:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for _ in range(20):
    sim.reset()
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.save_episode()        # flush this rollout as one episode
sim.stop_recording()          # flushes any trailing rollout automatically
# -> 20 episodes, each with its own episode_index / length / from_index / to_index
```

`save_episode` is idempotent on an empty buffer, so it is safe to call
unconditionally inside a loop. LeRobot computes `stats.json` per episode and then
aggregates, so per-rollout boundaries keep dataset statistics correct across the
`reset()` teleport between rollouts.

`reset()` is itself an episode boundary during a recording: if frames are
buffered when you call it, `reset()` flushes them as their own episode before
re-initializing the world (it reports the saved episode in its result text).
This means a bare `run_policy` + `reset` collection loop already produces one
episode per rollout - the explicit `save_episode()` is optional when you reset
between rollouts:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for _ in range(20):
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.reset()               # flushes this rollout as one episode, then resets
sim.stop_recording()          # flushes any trailing rollout automatically
# -> 20 episodes
```

Without either an explicit `save_episode()` or a `reset()` between rollouts, all
20 rollouts append to the same buffer and `stop_recording` flushes them as a
single `episode_index=0` (1200 steps in one episode). To DISCARD a partial
rollout instead of flushing it on the next `reset()`, call
`clear_episode_buffer()` first.

## Recording paths

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `save_episode` | `[lerobot]` | Close current rollout as one episode (call once per `run_policy` for N episodes) |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

## DatasetRecorder direct API

```python
from strands_robots.dataset_recorder import DatasetRecorder

recorder = DatasetRecorder.create(
    repo_id="user/my_dataset",
    fps=30,
    robot_type="so100",
    # When recording from a real LeRobot hardware robot pass the schema dicts
    # straight through:
    #   robot_features=robot.observation_features,
    #   action_features=robot.action_features,
    # When recording from a sim Robot (no `observation_features` attr), pass
    # `joint_names=[...]` instead - the recorder builds the schema for you.
    camera_keys=["default"],
    joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    task="pick up the red cube",
    # root=None → ~/.strands_robots/datasets/
    # vcodec="libsvtav1", streaming_encoding=True, image_writer_threads=4
)

for step in control_loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
recorder.save_episode()
recorder.finalize()
recorder.push_to_hub(tags=["so100", "sim"], private=False)
```

Append to existing dataset (requires `lerobot>=0.5.2`):

```python
recorder = DatasetRecorder.resume(repo_id="user/my_dataset", task="pick up the blue cube")
recorder.add_frame(observation, action)
recorder.save_episode()
recorder.finalize()
```

## Instance methods

| Method | What |
|--------|------|
| `add_frame(observation, action, task=None, camera_keys=None)` | Append one timestep |
| `save_episode()` | Flush buffer as a new episode |
| `clear_episode_buffer()` | Discard current episode |
| `finalize()` | Write metadata, stats, close writers |
| `push_to_hub(tags=None, private=False)` | Upload to a versioned HF dataset repo |
| `sync_to_bucket(bucket, run_id=None, private=True)` | Sync to a mutable HF Storage Bucket (`hf://buckets/...`) — Xet-deduped collection target; needs the `hf` CLI. `bucket` (`name` or `org/name`) and `run_id` (single segment) are allowlist-validated (`[A-Za-z0-9._-]`, no traversal) before the sync |

## Read back

Fully materialized (downloads everything):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds), ds[0].keys())
```

## Stream back (no full download)

`sim.stream_dataset()` is the in-process counterpart to `start_recording` /
`stop_recording` — it reads frames lazily from the Hub (or a local `root`) via
LeRobot's `StreamingLeRobotDataset`. Camera frames are decoded on the fly from
the MP4 shards; state/action come from the parquet shards.

```python
from strands_robots import Robot

sim = Robot("so100")
reader = sim.stream_dataset(
    "user/my_dataset",                 # or a local repo_id + root=
    root="/tmp/my_dataset",
    delta_timestamps={                 # optional: stacked time windows + *_is_pad masks
        "observation.state": [-0.0667, -0.0333, 0.0],
        "action": [0.0, 0.0333, 0.0667],
    },
    shuffle=False,                     # chronological for replay/eval
)
print(reader.num_episodes, reader.num_frames, reader.fps)
for frame in reader:
    ...

# torch DataLoader (shuffles INTERNALLY — do not pass shuffle=True):
for batch in reader.dataloader(batch_size=64, num_workers=4):
    ...
```

Equivalently, the standalone reader: `from strands_robots import StreamingDatasetReader`.

Useful kwargs (forwarded to `StreamingLeRobotDataset`, version-tolerant):
`episodes=[...]` (subset without download), `buffer_size`, `max_num_shards`,
`return_uint8=True` (default; halves frame bandwidth), and
`drop_videos=True` (proprio-only — skips video decode entirely, so it works on
edge devices without a torchcodec wheel).

For **training**, the upstream trainer uses the same engine:

```bash
python -m lerobot.scripts.train policy=act \
  dataset.repo_id=user/my_dataset dataset.streaming=true num_workers=4
```

> **macOS:** video streaming needs Homebrew ffmpeg on the dyld path. `import
> strands_robots` auto-fixes this (zero-touch); disable with
> `STRANDS_ROBOTS_NO_DYLD_SHIM=1`. See the README "Recording & streaming
> datasets" section.

## See also

- [Training](training/overview.md) - what to do with the data.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) - upstream spec.
