---
description: Prove a recorded dataset holds the episodes, pixels and columns it claims.
---

# Verify a dataset

An agent narrating "20 episodes recorded" is not proof: a single `run_policy(n_episodes=1)`, or 20
looped tool calls into one open buffer, produces one merged `episode_index=0` mega-episode while the
caller believes it recorded 20. Verify against the on-disk metadata instead.

```python
sim.stop_recording()
result = sim.verify_dataset_episodes(expected=20)
assert result["status"] == "success"   # else MISMATCH, fail loud
```

It reads two independent sources of truth and requires them to **agree**: the distinct
`episode_index` set in `meta/episodes/**/*.parquet` (the ground truth) and the `total_episodes`
header in `meta/info.json`. `status` is `"error"` when the parquet count differs from `expected` **or**
when the two disagree (`sources_agree` is then `False`), so a dataset matching `expected` on one
source alone still fails. The `{"json": {...}}` block carries `expected`, `actual`,
`info_total_episodes`, `info_problems`, `sources_agree`, `episode_indices` and `total_frames` for CI
gating; `read_dataset_episode_indices(root)` exposes the same facts in pure pyarrow, with no
`LeRobotDataset` instantiated.

A header that is present but is not a count at all - `2.5`, `"2"`, `true`, or a number outside double
range such as `1e400` - is a third outcome, distinct from a match and from an absent header:
`info_total_episodes` is `None`, the reason lands in `info_problems`, and `sources_agree` is `False`.
It is never coerced to a nearby number, because `int(2.5)` is `2` - exactly the count a two-episode
parquet holds, so coercing would certify the inconsistent dataset. Every reader of that header shares
one domain, `strands_robots.utils.declared_count`, so one file cannot get two verdicts.

## From the shell

```bash
strands-robots verify-dataset /path/to/dataset --expected 20   # exit 0 pass, 1 fail
strands-robots verify-dataset /path/to/dataset --json          # machine-readable report
strands-robots verify-dataset /path/to/dataset --no-check-videos  # skip the per-episode MP4 checks
```

The programmatic form is
`strands_robots.verify_dataset.verify_dataset(root, expected=None, min_frames=1, check_videos=True, check_stats=True)`,
returning the same report dict.

| Failure mode | What it catches |
|---|---|
| the mega-episode | fewer distinct episodes than `--expected` |
| header drift | `meta/info.json` `total_episodes` / `total_frames` differing from the parquet ground truth, or declaring something that is not a count - caught even without `--expected` |
| a short episode | any episode below `--min-frames` (default 1) |
| no pixels | a per-episode video file missing or empty on disk, resolved from `info.json`'s `video_path` template and the parquet's `chunk_index` / `file_index`, and counted in `video_files_checked` (skip with `--no-check-videos`) |
| a dead control column | `action` or `observation.state` written as all zeros because the writer's keys never resolved, read from the per-episode `min`/`max` stats LeRobot v3 writes inline - no video decode, no `data/` scan (skip with `--no-check-stats`) |

The video and dead-column checks are the modality siblings of the mega-episode: a dataset can carry
the right episode count and still have no pixels, or correct counts and pixels with a proprioceptive
column that never moved.

A multi-robot recording is graded one level finer. `start_recording` namespaces every declared name
with the robot's instance name (`alice__shoulder_pan`), so a resolution failure affecting one robot
leaves that robot's whole block zero while the other's carries measurements - the vector as a whole
still varies, and a whole-vector test reports `[PASS]`. The check splits the vector into the
per-robot blocks `meta/info.json` declares and names the offender:

```
feature 'observation.state' is identically zero for every 'bob' column across
episode 0 (20 frame(s)) - dead control column block
```

A zero *subset* of one robot's block is left alone (a gripper parked at zero for a whole episode is a
measurement), and a dataset declaring no column names is graded by the whole-vector rule.

`--expected` and `--min-frames` are non-negative integers, and each has a meaningful `0`:
`--expected 0` asks that a dataset be empty, `--min-frames 0` skips the length check. Anything else -
negative, fractional, non-finite - is reported and exits non-zero rather than applied, because a
value that is not a usable count would otherwise switch the check off and certify a dataset holding a
zero-length episode. The length check also runs only when the parquet carries per-episode lengths at
all, and availability is whether a `length` was *read*, not whether one was positive: three episodes
of zero frames are graded and named (`3 episode(s) below min_frames=1`), while a column that is
absent or wholly null stays unknown rather than zero.

`verify-dataset` always produces a report - it never crashes on the corruption it exists to flag. A
corrupt or foreign `meta/episodes` parquet, a non-v3 `video_path` template, or a truncated MP4 is a
problem string in the report and a non-zero exit code, not a traceback. Corruption confined to some
parquet shards (the usual outcome of an interrupted rsync or hub download) is localised: each
unreadable shard is named, the readable ones still supply `total_episodes` / `frames_per_episode`, and
the info.json, video and dead-column checks still run. Only a `meta/episodes` tree with no readable
shard reports zero episodes. `verify_dataset_episodes` additionally refuses to certify a dataset with
unreadable shards even when the readable count matches `expected` - the count is then a lower bound,
reported in `unreadable_files`.

## Incomplete recordings report themselves

`DatasetRecorder` is fail-fast by default (`strict=True`): a failed `LeRobotDataset` write raises
`strands_robots.recording_errors.RecordingFrameError`, and under `run_policy` that ends the rollout
with `status="error"` naming the frame the recording stopped being complete at. Continuing past a
lost frame is not a smaller failure - timestamps are positional, so the survivors are re-stamped into
a shorter span than they were captured over, and a rollout losing every other frame at 50 Hz yields
an episode labelled at 2x speed with no gap to detect.

`strict=False` trades that for best-effort recording: a failed write is counted in
`dropped_frame_count` and warned about at `WARNING` on the 1st, 2nd, 4th, 8th ... failure, so a 50 Hz
loop cannot flood the log. `stop_recording` is where those counts reach the caller and the last
chance to see them - it releases the recorder as it returns:

| Outcome | What `stop_recording` reports |
|---|---|
| some writes failed | success (`strict=False` chose to complete) naming the shortfall in the text and in `dropped_frame_count` beside `frame_count`: `the dataset holds 10 of the 20 frames recorded` |
| every write failed | error naming *that* reason - `all 20 frame(s) the recorder was fed failed to write, so the dataset holds 0 frames` - rather than the empty-dataset recipe the caller had just followed |

A failed `save_episode` is worse than a lost frame, not milder: the LeRobot episode buffer is
undefined after a partial write, so the recorder marks itself closed and `add_frame` then returns
immediately - no frame, no `RecordingFrameError`, no `dropped_frame_count`. Every flush therefore
refuses rather than continues. `save_episode()` and `stop_recording()` drop the poisoned recorder and
return `status="error"`; `run_policy(n_episodes=N)` aborts its remaining episodes (the facade and the
tool alike, reporting `recording_save_error` beside its parquet-truth counts); `reset()` surfaces the
failure instead of resetting into an undefined state; and a recorded `eval_policy` /
`evaluate_benchmark` stops at the episode whose flush failed:

```python
result = sim.eval_policy(robot_name="so100", n_episodes=20)
payload = next(b["json"] for b in result["content"] if "json" in b)
if payload["recording_save_error"]:      # None on every healthy evaluation
    ...   # status is "error"; episodes_completed is the episode it stopped at
```

`episodes_completed` and `success_rate` then cover only the episodes that ran, so an aggregate is
never reported over episodes whose frames reached no dataset. Those frames are labelled with the
instruction the *policy* was given - the caller's `instruction=`, else the benchmark's own
`spec.instruction` - the precedence `run_policy(instruction=...)` already has over the session's
`start_recording(task=...)`.

## See also

- [Recording & datasets](../recording.md) - the session-level verbs.
- [DatasetRecorder API](dataset-recorder.md) - the writer and the domains it refuses.
- [Episode labels](episode-labels.md) - a VLM judge over recorded episodes.
