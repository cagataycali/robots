---
description: Put a recorded dataset on the Hub or an HF Storage Bucket and read frames back without downloading all of it.
---

# Stream and sync

At the end of this page a recorded dataset lives somewhere other than one laptop's disk (a Hub repo, or an HF Storage Bucket as a mutable dump) and you can read frames back from either without downloading all of it, in a notebook, an eval loop or a DataLoader.

Both directions need the `[lerobot]` extra (lerobot 0.6.1 or newer for bucket streaming, `BUCKET_STREAMING_MIN_LEROBOT`) and `hf auth login`.

```python title="sketch"
from strands_robots.streaming_dataset import stream_dataset

reader = stream_dataset("you/so101_reach", episodes=[0, 1], buffer_size=1)   # buffer_size=1 is what delivers capture order; shuffle=False alone still reorders through the reservoir
for frame in reader:                            # dicts: observation.state, action, observation.images.front, timestamp
    print(frame["action"])
    break
loader = reader.dataloader(batch_size=64)       # a torch DataLoader over the same stream
```

## Two places to put a dataset

| target | when | how |
|---|---|---|
| Hub dataset repo (`push_to_hub`) | a finished dataset you will version and share | `sim.stop_recording(push_to_hub=True)`, or `DatasetRecorder.push_to_hub(tags=, private=)`. git-LFS history: every push adds |
| HF Storage Bucket (`sync_to_bucket`) | collection in progress, daily re-sync, a directory that keeps growing | `sim.stop_recording(bucket="you/collection", run_id="2026-09-27")`, `DatasetRecorder.sync_to_bucket(...)`, or `sync_dataset_to_bucket(root, bucket, run_id)` on any finalized directory |

A bucket is Xet-deduplicated: a re-sync uploads only changed chunks. It needs the `hf` CLI with `buckets` and `sync` (`huggingface_hub>=1.5`). The destination is `hf://buckets/<bucket>/<run_id>`.

`bucket` is `name` or `org/name` and `run_id` is one path segment. Both reach the `hf` argv from an agent-callable action, so both are checked against an allowlist first; a shell metacharacter, `..` or an extra separator is refused before any subprocess runs. `delete=True` forwards `--delete` (mirror semantics); `create=True` creates the bucket, `private=True` by default.

```python title="sketch"
from strands_robots.dataset_transfer import sync_dataset_to_bucket

print(sync_dataset_to_bucket("~/.cache/huggingface/lerobot/you/so101_real", "you/collection", run_id="arm-a-day3"))
# {'status': 'success', 'bucket_uri': 'hf://buckets/you/collection/arm-a-day3'}
```

The transfer needs no live recorder, sim world or lerobot import: a dataset `lerobot-record` wrote on hardware syncs the same way.

## Reading back

`StreamingDatasetReader.open(repo_id, ...)` (alias `stream_dataset`) wraps lerobot's `StreamingLeRobotDataset`. `sim.stream_dataset(repo_id)` does the same from a simulation session. Arguments:

| argument | meaning |
|---|---|
| `root=` | a local directory; overrides the Hub. A directory a recorder registered under `repo_id` in this process is used when `root` is not given |
| `episodes=[...]` | distinct non-negative indices; an index the dataset does not hold is refused rather than streamed as an empty read. Pair with `filter_episodes` from [label and judge](label-and-judge.md) |
| `delta_timestamps={key: [offsets]}` | stack past or future frames per key; checked against the fps grid unless `validate_deltas=False` |
| `image_transforms=` | applied to every image tensor |
| `tolerance_s=1e-4`, `revision=`, `buffer_size=1000`, `max_num_shards=16`, `seed=42`, `shuffle=True`, `return_uint8=True` | forwarded to lerobot |
| `drop_videos=True` | state-only stream; skips video decode entirely |

Video decode from a remote dataset needs `torchcodec` and `av`, which `[lerobot]` brings in as `lerobot[dataset]`; when they are missing the reader warns before the first frame instead of failing inside a worker.

## Streamed training

Training does not need this module: `python -m lerobot.scripts.lerobot_train --dataset.repo_id=you/so101_reach --dataset.streaming=true` uses `StreamingLeRobotDataset` through lerobot's own factory ([training](../training/lerobot.md)).

## The loop

Record on the sim or the arm, `verify-dataset --expected N`, label with the judge, `filter_episodes`, then stream exactly those episodes into training. Every step reads the parquet, not a counter.
