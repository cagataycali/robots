---
description: Replay a recorded episode in sim, stream a dataset without downloading it, train on it.
---

# Read back & replay

Fully materialized (downloads everything):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds), ds[0].keys())
```

## Replay an episode

`sim.replay_episode(repo_id, robot_name=..., episode=0, root=None, speed=1.0)` plays a recorded
episode back through the sim: each recorded frame is one control step, applied via `send_action` and
integrated for a full control period derived from the dataset fps, so a position-servo robot
reproduces the recorded trajectory. `speed` scales only the wall-clock playback rate. `root` resolves
from `repo_id` exactly as recording resolves it, so whatever id `start_recording` was given replays
with nothing restated:

```python
sim.start_recording(repo_id="sim_recording", task="pick the cube", fps=30)
...
sim.stop_recording()
sim.replay_episode("sim_recording", robot_name="so101")   # same id, same directory
```

`robot_name` follows the rule `run_policy`, `eval_policy` and `evaluate_benchmark` share: omit it in a
sole-robot scene, name it in a scene holding several (an omitted name there is refused with the
candidate list rather than replayed onto the first robot), and a name the scene does not hold - the
empty string included - is reported by name.

Each recorded action index is bound to an action key. By default those are
`robot_action_keys(robot_name)`, the actuator keys the recorder writes the `action` column in. Pass
`action_key_map` only when the dataset's ordering differs:

```python
sim.replay_episode(
    "user/my_dataset",
    robot_name="so101",
    root="/tmp/my_dataset",
    action_key_map=["1", "2", "3", "4", "5", "6"],  # one key per action index
)
```

`action_key_map` must be a non-empty list/tuple of unique strings whose length equals the recorded
action vector's width; a bare string (consumed one key per character), a non-string entry, a duplicate
key or a width mismatch is rejected before the dataset is fetched, never truncated to fit.

A `"success"` status means at least one recorded action reached the actuators and **every** frame that
carried one was applied; `frames_with_action` reports how many of `frames_applied` commanded the robot
rather than only advancing physics.

| Failure | Status | What the report names |
|---|---|---|
| a mapped key resolves to no actuator | `error` | the frame index it aborted at, `frames_applied`, and `unresolved_keys` |
| the episode's frames carry no `action` column | `error` | `frames_with_action: 0` and `recorded_columns`, the columns the frames do carry |

Both are refusals rather than a tolerated no-action replay: every frame would advance physics while
commanding nothing, and `Frames: N/N` would read exactly like a replay that worked.

## Stream back (no full download)

`sim.stream_dataset()` reads frames lazily from the Hub (or a local `root`) via LeRobot's
`StreamingLeRobotDataset` - camera frames decoded on the fly from the MP4 shards, state and action
from the parquet shards. The standalone reader is `from strands_robots import
StreamingDatasetReader`.

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

# torch DataLoader (shuffles INTERNALLY - do not pass shuffle=True):
for batch in reader.dataloader(batch_size=64, num_workers=4):
    ...
```

A `repo_id` that is itself a path (no `owner/name` slash, or `./`-prefixed) resolves to the directory
recording wrote to, so `sim.stream_dataset("sim_recording")` reads `./sim_recording`, not the Hub.

`shuffle` is **not** the read-order knob. It selects only which generator drives the reordering (one
reseeded from `seed` on every exhaustion, or the dataset's advancing one), so it decides
reproducibility across epochs - lerobot documents it as "whether to shuffle the dataset across
exhaustions". `StreamingLeRobotDataset` reorders either way: it samples a shard at random per frame and
yields from a reservoir buffer. The knobs above are what deliver the recorded order.

| Kwarg | Domain | Notes |
|---|---|---|
| `episodes=[...]` | list of episode indices | a subset without downloading the rest |
| `buffer_size`, `max_num_shards` | positive ints | `1` and `1` read in capture order |
| `tolerance_s` | non-negative finite | `0` requires an exact delta-grid match; `inf` used to switch the grid check off |
| `seed` | int | `0` is a seed, not "unset" |
| `return_uint8` | boolean, default `True` | halves frame bandwidth; `None` streamed float32 at ~4x |
| `drop_videos` | boolean | proprio-only, skipping video decode entirely - works on edge devices with no torchcodec wheel. Requires `delta_timestamps` with at least one non-video key, else `open()` raises |
| `streaming`, `shuffle`, `validate_deltas` | boolean | `validate_deltas=False` skips the delta-grid check |

The numeric kwargs are checked before the lerobot import, because `StreamingLeRobotDataset` validates
only `repo_type` and stores the rest verbatim - so `open()` raises `ValueError` naming the parameter
rather than opening a reader that streams zero frames (`max_num_shards=0`) or raising out of NumPy
part-way through iteration (`buffer_size=0`). The booleans are checked there too, on the same domain
the recording postures use: read by truthiness, each selected the branch the caller was opting out of.

```python
reader = sim.stream_dataset("user/d", drop_videos="false")  # ValueError: drop_videos must be a boolean
reader = sim.stream_dataset("user/d", validate_deltas=0)    # same refusal
```

Every kwarg is forwarded unconditionally: each lerobot-bearing extra floors lerobot at `0.6.1`, whose
`StreamingLeRobotDataset` accepts all of them, `repo_type` included. `open()` therefore never drops a
keyword to suit an older constructor - which for `repo_type` would have streamed the versioned dataset
namespace instead of the requested bucket, a different storage system. A below-floor lerobot gets
lerobot's own `TypeError` naming the keyword.

## Train on it

The upstream trainer uses the same engine:

```bash
lerobot-train --policy.type=act \
  --dataset.repo_id=user/my_dataset --dataset.streaming=true --num_workers=4
```

(`lerobot-train` is the entry point over `python -m lerobot.scripts.lerobot_train`; flags are draccus
`--dotted.key=value` form.)

> **macOS:** video streaming needs Homebrew ffmpeg on the dyld path. `import strands_robots`
> auto-fixes this for script runs; in a REPL, Jupyter or `python -c` it cannot re-exec, so the first
> video-decoding `stream_dataset` warns with the `export` line instead. Disable with
> `STRANDS_ROBOTS_NO_DYLD_SHIM=1`.

## See also

- [Recording & datasets](../recording.md) - produce the dataset.
- [Verify a dataset](verifying-datasets.md) - prove it holds what it claims.
- [Training](../training/overview.md) - what to do with the data.
