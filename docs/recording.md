---
description: DatasetRecorder - LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
---

# Recording & datasets

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_recording()
# LeRobot v3 dataset written to $HF_LEROBOT_HOME/user/my_dataset
```

`start_recording` requires `[lerobot]`. Without it, use `start_cameras_recording`
for plain MP4. The dataset format, its `meta/info.json`, stats and video layout
are LeRobot's - see the
[LeRobot dataset docs](https://huggingface.co/docs/lerobot). This page covers
what strands adds.

## `fps` must equal the rollout's `control_frequency`

The recorder writes one frame per control step and never decimates, and
LeRobot derives every timestamp from the declared `fps`
(`timestamp = frame_index / fps`), so a differing rate can only mislabel the
episode. `run_policy`, `PolicyRunner.run`, `start_policy` and
`start_recording` all refuse the mismatch before a frame is written:

```python
sim.start_recording(repo_id="user/my_dataset", task="t", fps=30)
sim.run_policy(robot_name="so100", policy_provider="mock")   # default 50.0 Hz
# -> "run_policy: the active recording declares 30 fps but this rollout captures
#     at control_frequency=50 Hz. [...] a 1.667x distortion of the episode
#     duration [...] Align the two rates: pass control_frequency=30 to
#     run_policy(), or record at the rollout's rate"
```

## Selecting which cameras to record

Every camera in the scene is recorded by default, including the implicit
`default` free camera. A policy with a fixed image schema (SmolVLA expects
exactly `observation.images.camera1/camera2/camera3`) needs the list pinned;
`cameras=` on `start_recording` (or `dataset_cameras=` on `run_policy`) refuses
a name that is not a camera:

```python
sim.add_camera(name="camera1", ...)
sim.add_camera(name="camera2", ...)
sim.add_camera(name="camera3", ...)
sim.start_recording(
    repo_id="user/my_dataset", task="pick up the cube", fps=50,
    cameras=["camera1", "camera2", "camera3"],   # drops the implicit 'default'
)
```

### Where the dataset is written (`root` / `overwrite`)

`root` is used verbatim. Without it an `owner/name` `repo_id` records into
`$HF_LEROBOT_HOME/{repo_id}` (default `~/.cache/huggingface/lerobot`) and a
path-like `repo_id` is the directory. An existing non-empty target is refused
unless `overwrite=True`; an existing dataset is appended to at its on-disk
`fps` (a different `fps` is refused, naming both rates). `overwrite` must be a
real boolean - `"false"` is refused, not read as truthy.

## Multi-episode recording

A recording session is one dataset. `run_policy(n_episodes=N)` runs N rollouts,
flushes an episode boundary after each and resets the sim between them. The
manual loop is the same three verbs - `reset()` is an episode boundary on every
backend, so either order works:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
for _ in range(20):
    sim.reset()
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.save_episode()        # flush this rollout as one episode
sim.stop_recording()          # flushes any trailing rollout automatically
```

## Verifying episode count

Twenty looped `run_policy` calls into one open buffer without `save_episode`
produce one merged `episode_index=0`. Verify against `meta/info.json`, not the
agent's narration:

```python
sim.stop_recording()
result = sim.verify_dataset_episodes(expected=20)
assert result["status"] == "success"   # else MISMATCH, fail loud
```

```bash
strands-robots verify-dataset /path/to/dataset --expected 20   # exit 0 pass, 1 fail
```

## Recording paths

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `save_episode` | `[lerobot]` | Close current rollout as one episode (call once per `run_policy` for N episodes) |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

`stop_cameras_recording` returns `status: "error"` when nothing was encoded
(the loop is still capturing, or no `imageio-ffmpeg`); the frames are still
held and a second call flushes them. A new cameras recording is refused while
the old one is still registered.

## Video codec (H.264 default, AV1 opt-in)

`start_recording` and `DatasetRecorder.create` / `resume` default to
`vcodec="h264"`, which every downstream reader - including
`cv2.VideoCapture` - decodes. Pass `vcodec="av1"` only when the whole
pipeline reads AV1.

## DatasetRecorder direct API

```python
from strands_robots.dataset_recorder import DatasetRecorder
recorder = DatasetRecorder.create(
    repo_id="user/my_dataset",
    fps=30,
    robot_type="so100",
    # hardware: robot_features=robot.observation_features, action_features=robot.action_features
    # sim: joint_names=[...] and the recorder builds the schema; the names are the
    # observation's own keys (`list(sim.get_observation()["so100"].keys())`)
    camera_keys=["default"],
    joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    task="pick up the red cube",
)
for step in control_loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
recorder.save_episode()
recorder.finalize()
recorder.push_to_hub(tags=["so100", "sim"], private=False)

# append to an existing dataset (lerobot>=0.5.2)
recorder = DatasetRecorder.resume(repo_id="user/my_dataset", task="pick up the blue cube")
```

`add_frame` raises when a declared joint or action name is missing from the
frame; nothing is recorded as a stand-in `0.0`. `create()` refuses before it
touches disk: a failed lerobot import (the
`ImportError` names which of `strands-robots[lerobot]` / `lerobot[dataset]` /
a version-pinned reinstall fixes it); duplicate or blank `camera_keys` /
`joint_names` / `action_names`; a `camera_dims` entry that is not a declared
camera or not a `(height, width)` pair (camera features are written
`(height, width, 3)`, names `[height, width, channels]` - lerobot's own
layout); a non-positive `video_width` / `video_height`; an `fps` that is not a positive whole number; a non-boolean
`use_videos` / `streaming_encoding` / `overwrite`; an existing dataset
directory without `overwrite=True`.

The recorder is fail-fast (`strict=True`): a frame LeRobot cannot write raises
`RecordingFrameError` and ends the rollout, and an episode it cannot flush
closes the recorder and stops a recorded `eval_policy` with
`recording_save_error` set in the result.

## Instance methods

| Method | What |
|--------|------|
| `add_frame(observation, action, task=None, camera_keys=None)` | Append one timestep |
| `save_episode()` | Flush buffer as a new episode |
| `clear_episode_buffer()` | Discard current episode |
| `finalize()` | Write metadata, stats, close writers |
| `push_to_hub(tags=None, private=False)` | Upload to a versioned HF dataset repo (`private` must be a boolean) |
| `sync_to_bucket(bucket, run_id=None, private=True)` | Sync to a mutable HF Storage Bucket (`hf://buckets/...`) |

## Replay an episode

`sim.replay_episode(repo_id, robot_name=..., episode=0, root=None, speed=1.0)`
plays a recorded episode back through the sim: each frame is one control step
applied via `send_action` for a control period derived from the dataset `fps`.
A recording made with a path-like `repo_id` replays from the same directory.
`action_key_map=[...]` names one actuator per action index; an unresolvable key
returns `status: "error"` with `unresolved_keys` and `frames_applied: 0`.

## Stream back (no full download)

`sim.stream_dataset()` reads frames lazily from the Hub or a local `root`
through LeRobot's `StreamingLeRobotDataset`; `buffer_size=1` is the knob that
delivers capture order (the reservoir otherwise reorders), and the reader's
`dataloader()` shuffles internally:

```python
from strands_robots import Robot
sim = Robot("so100")
reader = sim.stream_dataset(
    "user/my_dataset",                 # a path-like repo_id needs no root=
    root="/tmp/my_dataset",
    delta_timestamps={                 # optional: stacked time windows + *_is_pad masks
        "observation.state": [-0.0667, -0.0333, 0.0],
        "action": [0.0, 0.0333, 0.0667],
    },
    buffer_size=1,                     # capture order for replay/eval:
    max_num_shards=1,                  # one reservoir slot, one shard
)
print(reader.num_episodes, reader.num_frames, reader.fps)
for frame in reader:
    ...
# torch DataLoader (shuffles INTERNALLY — do not pass shuffle=True):
for batch in reader.dataloader(batch_size=64, num_workers=4):
    ...
```

Boolean kwargs (`drop_videos`, `validate_deltas`, `return_uint8`, `streaming`)
must be real booleans and are refused otherwise. `repo_type="bucket"` requires
`lerobot>=0.6.1`, which the `[lerobot]` extra floors; an older lerobot raises
`RuntimeError` naming the upgrade. Training streams through the same engine:

```bash
python -m lerobot.scripts.lerobot_train --policy.type=act \
  --dataset.repo_id=user/my_dataset --dataset.streaming=true --num_workers=4
```

> **macOS:** video streaming needs Homebrew ffmpeg on the dyld path. `import
> strands_robots` auto-fixes this; disable with `STRANDS_ROBOTS_NO_DYLD_SHIM=1`.

## See also

- [Training](training/overview.md) - what to do with the data.
- [Steerable annotation](data/annotation.md) - add language conditioning columns to a recorded dataset.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) - upstream spec.
