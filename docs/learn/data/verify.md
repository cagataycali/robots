# Verify

At the end of this page you can prove, from the parquet on disk and not from anyone's narration, that a recorded dataset holds the episodes you intended, with frames in every one, video files that match, and no dead control column.

This runs without lerobot; only `pyarrow` is needed. A toy dataset with three episodes:

```python
import json, pathlib
import pyarrow as pa, pyarrow.parquet as pq
from strands_robots.verify_dataset import verify_dataset

root = pathlib.Path("/tmp/toy_dataset")
(root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({"episode_index": [0, 1, 2], "length": [90, 90, 90]}),
               root / "meta/episodes/chunk-000/file-000.parquet")
(root / "meta/info.json").write_text(json.dumps({"total_episodes": 3, "total_frames": 270, "fps": 30, "features": {}}))

print(verify_dataset(root, expected=3)["status"])        # success
print(verify_dataset(root, expected=20)["problems"][0])
# expected 20 episode(s) but parquet holds 3 - the recording did not produce the intended number of distinct episodes
```

## The command

```bash
strands-robots verify-dataset ~/.cache/huggingface/lerobot/you/so101_reach --expected 5
```

```text
[PASS] /Users/you/.cache/huggingface/lerobot/you/so101_reach
  episodes (parquet): 5
  frames   (parquet): 450
  expected episodes : 5
  info.json episodes: 5
```

Exit code 0 on success, 1 on any problem. Flags:

| flag | meaning |
|---|---|
| `--expected N` | require exactly N distinct episodes, the number you intended to record |
| `--min-frames K` | every episode must hold at least K frames (default 1; `0` disables) |
| `--no-check-videos` | skip the per-episode video file checks |
| `--no-check-stats` | skip the dead-control-column check |
| `--json` | the report as JSON |

## What it checks

1. `meta/episodes/**/*.parquet` exists and holds at least one distinct `episode_index`. An empty directory reports "The dataset is empty or was never finalized (episodes are flushed to parquet at stop_recording/finalize)".
2. Every episode has at least `min_frames` frames.
3. `meta/info.json`'s `total_episodes` and `total_frames` agree with the parquet. A header that is not a count, or a file that is not a JSON object, is corrupt metadata, not a missing one.
4. With `--expected`, the parquet holds exactly N episodes. This is the mega-episode check: a collection run that buffered every frame into `episode_index=0` has the right frame total and the wrong episode count.
5. Every video file the dataset references (resolved from `info.json`'s `video_path` template and the episode parquet's `videos/<key>/chunk_index` and `file_index` columns) exists, is non-empty, and, when the container header can be read, holds exactly the frames the parquet maps into it.
6. No episode's `action` or `observation.state` column is identically zero. A multi-robot vector is split into the per-robot blocks `info.json` declares; a wholly zero block is flagged, a zero subset of one robot's block (a gripper parked for the whole episode) is a measurement and is not.

The parquet is the ground truth. The checker never trusts an agent's "recorded 20/20" or the recorder's in-memory counters.

## From the simulation

`sim.verify_dataset_episodes(expected=5)` runs the same reader against the session's own dataset root and is the right last line of a collection script. `read_dataset_episode_indices(root)` in `strands_robots.dataset_metadata` is the shared reader all three surfaces use (the sim facade, the CLI, the episode judge), so they cannot disagree.

## The report

`verify_dataset()` returns a dict: `status`, `ok`, `root`, `total_episodes`, `total_frames`, `episode_indices`, `frames_per_episode`, `expected`, `info_total_episodes`, `info_total_frames`, `video_files_checked`, `problems`. A partially corrupt dataset reports the truth of the readable files with each broken file named in `problems`.

Next: [label and judge](label-and-judge.md).
