---
description: DatasetRecorder's direct API, the domains it refuses, MP4 clips, codecs and publishing.
---

# DatasetRecorder API

The writer behind [`start_recording`](../recording.md). Drive it directly when there is no
`Simulation` or `HardwareRobot` in the loop - a bare control loop, a hardware script, a converter.

## Recording surfaces

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `save_episode` | `[lerobot]` | Close current rollout as one episode |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

`stop_cameras_recording` returns a verdict. Two things stop it finishing, and both answer with a
structured error carrying `stopped: False` and the per-camera buffered counts, no MP4 written, and
the recording left registered so a later call can still encode it:

- **A join that expires.** The capture thread is still inside `render` after 5 s (a wedged GL
  context, an EGL device that stopped answering), and reading a buffer it may still append to is
  refused rather than encoded.
- **A flush with no encoder.** `imageio` ships with `[sim-mujoco]` but declares the MP4 plugin
  `imageio_ffmpeg` as an extra of its own, so an install can have one and not the other. The flush
  needs both and quotes whichever is missing, before opening any writer.

Retrying the stop is the remedy in both cases; on a recording whose loop has exited it joins
immediately and encodes.

| `phase` | `running` | `thread_alive` | Meaning |
|---|---|---|---|
| `recording` | `True` | `True` | Capturing normally. |
| `stopping` | `False` | `True` | A stop's join expired; frames can still land. |
| `unflushed` | `False` | `False` | The loop has exited, frames still unencoded - stop again to flush. |
| `idle` | - | - | Nothing registered; there is no buffer left to encode. |

Only a flush deregisters a recording, so both start verbs refuse a new one for as long as the old is
registered - whether or not its thread is alive - and `[idle]` is a promise that nothing is pending.
Registration is published before the capture thread starts, so two racing starts cannot both proceed;
if the thread cannot start at all, the recording is deregistered again and the start reports it. The
Isaac backend captures through the `on_frame` hook `start_cameras_recording` returns rather than a
daemon thread, so it has no join to expire; its encoder rule is worded from the same place.

`fps`, `width`, `height` and `max_frames_per_camera` must be positive whole numbers - the domain
`run_policy(video={...})`, `start_recording(fps=...)` and `strands_robots.rendering.encode_clip`
share, so an unusable value is a structured error naming the parameter rather than a success that
writes no file. Omit `width`/`height` to use each camera's configured resolution. `encode_clip`
raises `ValueError` for a rate it cannot honor and `RuntimeError` when the encoder wrote no clip, so
a returned path always names a clip that exists; its `quality` is a finite number in `[1, 10]`,
higher being better (`0` and `True` are refused, a NumPy real is converted).

## Video codec (H.264 default, AV1 opt-in)

`start_recording` (and `DatasetRecorder.create`/`resume`) default to `vcodec="h264"`, which OpenCV's
`cv2.VideoCapture` - what most downstream VLM video readers use - can decode:

```python
import cv2
cap = cv2.VideoCapture(".../videos/observation.images.base/chunk-000/file-000.mp4")
ok, frame = cap.read()   # H.264: ok is True, frame is a real (H, W, 3) array
```

`vcodec="libsvtav1"` opts into AV1 for smaller files in storage-constrained pipelines. It reads back
fine through LeRobot's own loader (`torchcodec`/`pyav`), but OpenCV wheels commonly lack an AV1
decoder and silently yield 0 frames - so avoid it if anything downstream reads the videos through
OpenCV. Codec names (`h264`, `hevc`, `libsvtav1`) and ffmpeg encoder names (`libx264`, `libx265`) are
both accepted and normalized to the installed LeRobot's encoder config; an unsupported codec fails
loudly rather than reverting to the default.

## Direct API

```python
from strands_robots.dataset_recorder import DatasetRecorder

recorder = DatasetRecorder.create(
    repo_id="user/my_dataset",
    fps=30,
    robot_type="so100",
    # From a real LeRobot hardware robot, pass the schema dicts through:
    #   robot_features=robot.observation_features,
    #   action_features=robot.action_features,
    # From a sim Robot, pass `joint_names=[...]` instead and the recorder builds
    # the schema. The names must be the observation's own keys:
    # `list(sim.get_observation()["so100"].keys())`.
    camera_keys=["default"],
    joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    task="pick up the red cube",
    # root=None -> $HF_LEROBOT_HOME/user/my_dataset
    # vcodec="h264", streaming_encoding=True, image_writer_threads=4
)

for step in control_loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
recorder.save_episode()
recorder.finalize()
recorder.push_to_hub(tags=["so100", "sim"], private=False)
```

`DatasetRecorder.resume(repo_id="user/my_dataset", task="pick up the blue cube")` appends to the
directory `create()` wrote to (requires `lerobot>=0.5.2`) and resolves `root` by the same rule
recording does. Use it, not `create(overwrite=True)`, when the point is to add episodes.

## What `create()` refuses

Every check below runs in one guard block - before the lerobot extra is probed and before the
on-disk target is touched - so a refused call cannot delete a dataset on its way to being reported.

| Parameter | Domain | Refused because |
|---|---|---|
| `camera_keys`, `joint_names`, `action_names` | list of distinct non-blank names (`None` and `[]` mean "derive it") | a bare string is iterable per character, so `joint_names="gripper"` declared seven columns that each recorded `0.0`; a repeated name collapses where it keys a dict and doubles where it indexes a position |
| `camera_dims` | mapping of a **declared** camera to a `(height, width)` pair of positive ints - note the order, the reverse of the pair below | it is a declaration, not a resize: an undeclared key is never looked up, so its camera silently took the global pair, and a non-integer is written into `meta/info.json` as given, so no frame can match |
| `video_width` / `video_height` | positive ints; the shape of every camera `camera_dims` does not cover | same |
| `fps` | positive whole number | LeRobot rejects only `fps <= 0`, so `2.7`, `nan` and `inf` created a dataset that then saved zero frames, and `fps=True` recorded at 1 fps |
| `use_videos`, `streaming_encoding`, `overwrite`, `strict`, `private`, `delete` | boolean | read by truthiness, `"false"`/`"no"`/`"off"`/`"0"` select the branch the caller is opting out of - and `overwrite` guards a delete |
| an existing target | `overwrite=True` wipes it, an empty directory is cleared | `overwrite=False` on a dataset raises `FileExistsError` naming both routes; a non-empty non-dataset directory raises `ValueError` rather than deleting unrelated files |

The shape rules are the facades' rules by construction - one shared domain, reached from both - so a
value `start_recording` accepts cannot be refused deeper, and a rate that disagrees with a rollout's
`control_frequency` stays the facades' own check.

## Import errors name the install that fixes them

`create()` and `resume()` import `lerobot.datasets.lerobot_dataset`, which fails for unrelated
reasons needing different instructions, so the `ImportError` says which one happened:

| Cause | What the error says to do |
|-------|---------------------------|
| lerobot itself is absent | `pip install 'strands-robots[lerobot]'` |
| lerobot is installed, but a package its dataset stack needs (`datasets`, `pandas`, `pyarrow`, `av`, `torchcodec`) is not | `pip install 'lerobot[dataset]'` - installing lerobot alone does not pull those in |
| lerobot is installed but does not provide that module (an out-of-range or from-source lerobot) | `pip install 'strands-robots[lerobot]'`, which pins the supported range |
| the import failed with nothing missing (a binary conflict between installed packages) | No install fixes it; reconcile the conflicting packages |
| `torchcodec is installed but cannot load in this process; decoding video with pyav instead` | Nothing is broken - recording and read-back use pyav. `strands-robots doctor` has the full diagnosis |

## Instance methods

| Method | What |
|--------|------|
| `add_frame(observation, action, task=None, camera_keys=None)` | Append one timestep |
| `save_episode()` | Flush buffer as a new episode |
| `clear_episode_buffer()` | Discard current episode |
| `finalize()` | Write metadata, stats, close writers |
| `push_to_hub(tags=None, private=False)` | Upload to a versioned HF dataset repo; `private` selects the published visibility, so it is a boolean |
| `sync_to_bucket(bucket, run_id=None, private=True)` | Sync to a mutable HF Storage Bucket (`hf://buckets/...`), a Xet-deduped collection target. `bucket` (`name` or `org/name`) and `run_id` (single segment) are allowlist-validated (`[A-Za-z0-9._-]`, no traversal) before the sync; `delete=True` mirror-deletes remote files absent locally |

`sync_to_bucket` needs the `hf` CLI with the `buckets`/`sync` subcommands
(`pip install -U "huggingface_hub>=1.5"` + `hf auth login` - those subcommands first ship in 1.5.0;
every earlier release, 1.0-1.4.x included, installs an `hf` entry point without them). It is the
recorder's own method, so it needs the live session; any directory already on disk - recorded earlier
in the process, or on hardware via `lerobot-record` - syncs (or re-syncs daily) through the
module-level helper:

```python
from strands_robots import sync_dataset_to_bucket

sync_dataset_to_bucket("/tmp/demo", "your-org/robot-fave")
# -> {"status": "success", "bucket_uri": "hf://buckets/your-org/robot-fave/demo"}
```

`run_id` defaults to the directory name; pass `run_id="nightly"` to choose the bucket subpath, and
`delete=True` for mirror semantics.

## See also

- [Recording & datasets](../recording.md) - the session-level verbs.
- [Verify a dataset](verifying-datasets.md) - what a failed write or flush reports.
