---
description: Label recorded episodes with a VLM judge layered on deterministic predicate verdicts - quality grades, failure-mode tags, and dispute annotations in a schema-versioned sidecar, then train on the filtered subset.
---

# Episode labels (judge on top of predicates)

Episode validation with `evaluate_benchmark` is deterministic-only: predicates
over simulator state score success and failure per the task definition. That
covers the objective checks, but not what a human reviewer looks at in
recorded episodes - smooth vs jerky motion, near-misses, camera occlusion,
wrong-but-lucky successes. For automated episode farming, that labeling step
is the remaining human bottleneck.

`strands_robots.episode_labels` closes it with a **two-stage verdict**, the
same doctrine as the safety and dispatch layers:

1. **Deterministic predicates run first and are authoritative** for what they
   can measure. Their per-episode verdicts (from `evaluate_benchmark`, or from
   `run_policy(stop_when=...)` per-episode predicate stops) are recorded with
   `record_deterministic_verdicts`.
2. **A judge annotates on top** - a quality grade, a failure-mode tag from a
   fixed taxonomy, a free-text note. The judge can never overturn a
   deterministic verdict: `annotate_episode` (and the `write_label` tool)
   writes only the `judge` block, refuses an episode with no deterministic
   verdict, and records a disagreeing `success_opinion` as
   `disputes_verdict: true` while the verdict stands. Pinned by
   `tests/test_episode_labels.py` and `tests/tools/test_episode_judge.py`.

The judge's own tools and the agreement measurement that qualifies it are on
[The episode judge](episode-judge.md). End-to-end walkthrough:
`examples/17_judge_recorded_episodes.py` (record -> verdicts -> judge ->
agreement measurement -> filtered re-training).

## The sidecar

Labels live in `episode_labels.json` at the dataset root, next to LeRobot's
own `meta/` / `data/` / `videos/` - episode-level metadata, so training can
filter without rewriting a single parquet shard. The file is schema-versioned
(`schema_version: 1`); a version this build does not know is refused on read
rather than misread. Writes are two-phase (temp file + atomic rename).

Reading is the boundary for a sidecar this build did not write, so the same
refusal reaches the records: a verdict-bearing field outside its domain, or a
block that cannot state the thing it exists to state, is reported against the
file and the episode rather than reinterpreted. The descriptive fields
(`steps`, `cumulative_reward`, `seed`, `note`, `model`, `labeled_at`,
`success_opinion`) are carried through untouched: nothing branches on them, so
a surprising value there is a record to read rather than a verdict to refuse.

```json
{
  "schema_version": 1,
  "benchmark": "so100_pan_reach",
  "episodes": {
    "0": {
      "episode_index": 0,
      "deterministic": {
        "success": false,
        "failure": true,
        "steps": 15
      },
      "judge": {
        "quality": "low",
        "failure_mode": "incomplete",
        "note": "budget exhausted at 15 steps without reaching the target",
        "success_opinion": true,
        "disputes_verdict": true,
        "model": "scripted-heuristic",
        "labeled_at": 1755590400.0
      }
    }
  }
}
```

Field domains. Each holds in both directions - the writer refuses a value
outside it, and so does the reader, so a sidecar another writer produced is
held to the same domains as one this build wrote:

| field | domain |
|---|---|
| `episode_index` | a non-negative whole number, on the shared domain every surface applies. Holds in each spelling the index arrives in: the `episode` argument of `deterministic_verdict` / `annotate_episode` and the judge tools, an `episodes[i]["episode"]` entry handed to `record_deterministic_verdicts`, and a key of `measure_agreement`'s holdout mapping. A value outside it selects a *different* episode rather than failing slowly - `True` is `1` to an index - so it is refused and named |
| `quality` | `low` / `medium` / `high` (ordered; filters compare rank). An *execution* grade, orthogonal to the outcome - see below |
| `failure_mode` | `null` or one of `jerky_motion`, `near_miss`, `camera_occlusion`, `wrong_but_lucky`, `drift`, `collision`, `incomplete`, `other`. Holds in each spelling the tag arrives in: the `failure_mode` argument of `annotate_episode`, a `judge.failure_mode` a reader loads from the sidecar, and a holdout entry's `failure_mode` handed to `measure_agreement`. The judge's tag is confined to the vocabulary at both ends, so a tag outside it can never equal one - it would be counted as a disagreement and understate the calibration rather than measure it |
| `success_opinion` | `null` (no opinion) or a boolean |
| `disputes_verdict` | derived: opinion present and different from the deterministic `success` |
| `model` | free-form provenance (`"human"`, a model id, an endpoint) |

A `failure_mode` is deliberately legal on a deterministically *successful*
episode: `near_miss` and `wrong_but_lucky` are exactly the annotations that
make a success worth excluding from training data.

`quality` grades the **execution visible in the recording** - smoothness,
directness, control - never the outcome. The deterministic verdict already
carries success/failure and `filter_episodes` gates on that verdict, so with
`require_success=True` a grade only ever discriminates among successes, and a
jerky or lucky success graded `high` for succeeding is exactly the episode the
grade exists to exclude. An unsteered judge grades the outcome instead, so
`JUDGE_SYSTEM_PROMPT` states the contract where the model reads it (a clean
failure can be `medium` or `high`; a jerky or lucky success can be `low`), and
`measure_agreement` is where to confirm a given judge honours it.

## Filtering and re-training

`filter_episodes` selects on the deterministic verdict (authoritative - a
judge opinion never admits a failed episode) plus the judge's quality grade:

```python
from strands_robots.episode_labels import filter_episodes
from strands_robots.training import TrainSpec, create_trainer

chosen = filter_episodes("/data/pick_place", require_success=True, min_quality="medium")

trainer = create_trainer("lerobot_local", device="cpu")
spec = TrainSpec(
    dataset_root="/data/pick_place",
    base_model="",
    output_dir="/data/pick_place_ft",
    steps=2000,
    val_episodes=3,
    extra={"policy_type": "act", "dataset.episodes": chosen},
)
result = trainer.train(spec)
```

The subset reaches lerobot as `DatasetConfig.episodes` through the typed
`extra` passthrough; the dataset itself is untouched. The same list feeds the
read side: `stream_dataset(..., episodes=chosen)`.

`val_episodes` is sized against the episodes the run LOADS, not against the
dataset's `total_episodes`: the three reserved here come out of `chosen`, so a
filter that kept 15 of 30 episodes trains on 12 and validates on 3. Asking for
more episodes than the subset holds is refused before the run starts, naming
both counts.

## Relation to steerable annotation

[Steerable annotation](annotation.md) (`lerobot-annotate`) writes *frame-level
language columns* into the dataset's parquet shards for training
language-conditioned policies. Episode labels are the complementary layer:
*episode-level* quality/failure metadata in a sidecar, for deciding **which**
episodes to train on. The two compose - annotate the episodes the judge kept.

## See also

- [The episode judge](episode-judge.md) - the labeling tools and the calibration
  step before trusting them.
- [Verify a dataset](verifying-datasets.md) - the deterministic verdicts labels
  sit on top of.
