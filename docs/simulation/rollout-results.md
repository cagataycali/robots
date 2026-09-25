---
description: The json payload a rollout returns, and the async-RTC chunk pipeline it reports on.
---

# Rollout results

What a rollout hands back: the result payload, the action-health fields that
expose a crippled-but-successful run, and the chunk-prefetch telemetry. Running,
refusing and stopping a rollout is [Policy rollouts](rollouts.md).

## What the result reports

`run_policy` returns a `{"json": {...}}` block beside the `text`, mirroring
`eval_policy`:

| Field | Meaning |
|-------|---------|
| `robot_name`, `policy`, `instruction`, `n_steps` | what ran |
| `elapsed_s` | measured on a monotonic clock, so no date correction can move it |
| `stopped_early`, `stopped_reason` | `budget`, `predicate`, `stopped`, or an error |
| `actions_applied`, `steps_advanced` | actions that **commanded** the robot, and physics steps taken - an advanced step is not a commanded action, and an action dict naming no actuator commands nothing |
| `action_errors`, `action_resolution_rate`, `partial_action_failure_rate` | per-key resolution health, over the steps whose per-actuator credit is *known*: a coarse backend error, and a step keyed by driven joint names rather than actuators, are excluded rather than scored as misses, so an empty map with `0.0` means unknown, not undriven |
| `video_path` (`None` when no MP4 was written), `video_frames`, `video_fps` | `video_fps` is the rate the MP4 *plays* at - the requested `fps` capped to `control_frequency`, since a rollout renders at most one frame per control step |
| `sim_time_s` | when the backend reports it |
| `stop_when_true_at_reset`, `stop_when_reset_warning` | see [predicates](predicates.md) |
| the `chunk_prefetch_*` fields | see below |

`status` reflects whether the robot *moved*: a run where no step resolved any key,
or named one at all (`actions_applied: 0` - how a model declaring no `<actuator>`
block behaves until `actuate_robot` adds a position servo per joint), returns
`status="error"`, while a run where only *some* keys resolve is operational and adds
an `N/M action steps had unresolved keys` note. Both eval routes tolerate one empty chunk
per step and refuse the *aggregate* when `actions_applied` is zero, since `success_rate`
/ `avg_reward` / `pass_hat_k` would then describe the scene's initial state. A criterion that
*raises* is fatal on every route (`success_fn`, `is_success` / `is_failure`,
`stop_when`), naming the criterion, the episode and the step; verdicts are read with
`bool()`, so a `numpy.bool_` is accepted. `on_frame` is best-effort telemetry -
logged and survived - except for a `RecordingFrameError`, which is data loss.

At `n_episodes > 1` the aggregate adds `total_steps`, `stopped_reasons` (aligned
with `episodes`), `video_paths` and the per-episode `episodes` records, each
carrying its own action health, so the worst episode is
`max(e["partial_action_failure_rate"] for e in report["episodes"])`. It also keeps
what the one policy object reports: `positional_fallback_used` /
`generic_state_keys_used` / `missing_state_keys_used` and `policy_load_time_s` /
`policy_load_cache_hit` / `policy_resident_rss_mb`. Read the binding flags here
first - this is the shape that collects a dataset, and a `true` flag means those
episodes recorded a robot moving on meaningless inputs under `status="success"`. A
`policy_load_cache_hit` of `false` on episode 2+ means the policy was rebuilt
instead of reused via `policy_object=`.

`seed=` makes a single rollout reproducible: it reseeds Python / NumPy / torch /
cuDNN and forwards `policy.reset(seed=...)`, so a stochastic policy repeats its
trajectory on the same scene. `eval_policy` already seeds per episode.

`eval_policy` and `evaluate_benchmark` take the same `video={...}` config as
`run_policy` (`path` enables it, plus `fps` / `camera` / `width` / `height`; an
unknown key, a non-positive size, or one field spelled twice with two values is a
caller error) but write **one MP4 per episode**, `_ep{i}` inserted into the filename
(`eval.mp4` -> `eval_ep0.mp4`, ...) and listed in `video_paths`. The path is
validated and the camera probed up front, so a bad camera fails immediately rather
than after N episodes of empty MP4s, and frames are captured synchronously on the
eval thread, so recording does not perturb a bit-stable rollout.

`sim.register_builtin_benchmarks()` makes the shipped benchmarks appear in
`list_benchmarks()` and run via `evaluate_benchmark(...)` without hand-authoring a
spec; it is opt-in, so importing the library mutates no registry, and
`builtin_benchmark_specs()` returns the spec dicts to fork. It ships
`go2_walk_forward`: walk the Unitree Go2's base past `x = 2 m` (`base_beyond_x`),
fail on a topple (`base_tipped`) or a height collapse (`base_below_z`), shaped by a
dense `base_velocity_tracking` + `base_height` + `base_orientation` reward.

## Async-RTC chunk pipeline (latency masking)

`async_rtc` overlaps inference with execution: while the current chunk drains, the
*next* `get_actions` runs on one background worker (from a fresh mid-chunk
observation) and is swapped in atomically at the seam.

```
async_rtc=True (inference <= chunk execution):

chunk N exec   |####============|
prefetch N+1            |~~~~~~~|              <- fires at ~50% of chunk N
chunk N+1 exec                  |####========|   <- ready at the seam: HIT, no stall

async_rtc=False (synchronous chunk-then-drain):

chunk N exec   |####|
infer N+1            |~~~~~~~|                 <- the loop stalls here every seam
chunk N+1 exec               |####|
```

`async_rtc=None` (the default) resolves from `policy.is_chunk_emitting()`:
chunk-emitting VLA / flow-matching policies (pi0, pi0.5, pi0-FAST, SmolVLA,
MolmoAct2) get the overlap, single-step policies (MockPolicy, classical planners)
stay synchronous where overlap gains nothing; an explicit `True` / `False` wins.
`Policy.is_chunk_emitting()` defaults to `execution_horizon > 1`, and
`LerobotLocalPolicy` also reports `True` for an RTC model or a checkpoint driven via
`predict_action_chunk` (MolmoAct2) - see
[LeRobot Local -> RTC](../policies/lerobot-local.md#synchronous-vs-async-chunk-execution-in-sim).
An empty *prefetched* chunk degrades to one synchronous re-query before erroring, so
a transient hiccup does not kill a healthy rollout; a prefetch blocking at the seam
logs a starvation warning, and `rtc_inference_timeout_s` bounds a stuck inference -
the swap then errors with the telemetry below rather than waiting out every
remaining chunk.

| Field | Meaning |
|-------|---------|
| `chunk_prefetch_enabled` | Whether the overlap pipeline ran (the resolved `async_rtc`) |
| `policy_rtc_enabled` | The policy's own `supports_rtc` - real seam blending, independent of the pipeline |
| `chunk_prefetch_chunks_acquired` | Chunks acquired (cold start + swaps + re-queries), counted on the synchronous path too |
| `chunk_prefetch_hits` | Seams where the next chunk was already computed (stall hidden) |
| `chunk_prefetch_blocks` | Seams where the runner had to wait for inference (seam starved) |
| `avg_inference_ms` / `max_inference_ms` | Mean and slowest `get_actions` wall time |

A healthy masked rollout shows `chunk_prefetch_hits` near the chunk count and
`chunk_prefetch_blocks == 0`. The `rtc_async_enabled`, `rtc_chunks_acquired`,
`rtc_prefetch_hits`, `rtc_prefetch_blocks`, `rtc_avg_inference_ms` and
`rtc_max_inference_ms` spellings are still emitted with the same values for one
release.

`eval_policy` takes the same two knobs but defaults to `async_rtc=False`: the
synchronous eval pauses the world during inference, so the success rate is
bit-stable. `async_rtc=True` measures robustness to inference latency instead - the
prefetch feeds a staler mid-chunk observation, so the measured rate can shift. It is
rejected on the benchmark/spec path, which stays synchronous and declares an
observed delay of exactly `0` before every inference.

## See also

- [Policy rollouts](rollouts.md) - the actions, their parameter domains and stopping.
- [Predicates](predicates.md) - the `stop_when` and benchmark clauses a reason names.
- [Rollout observers](observers.md) - reading a rollout step by step instead of at the end.
- [Physics and actions](physics.md) - what an applied action writes.
