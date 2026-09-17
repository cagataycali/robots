---
description: Running, stopping, evaluating and watching a policy in simulation.
---

# Policy rollouts

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=50.0`, `action_horizon=8`, `n_steps=None`, `seed=None`, `async_rtc=None`, `rtc_inference_timeout_s=None`, `stop_when=None`, `observer=None` ([observers](observers.md)), `video=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""` - every rollout) |
| `list_policies_running` | - |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (optional; auto-resolves the sole robot), `n_episodes=1`, `max_steps=300`, `success_fn=None`, `async_rtc=False`, `rtc_inference_timeout_s=None`, `video=None` |
| `evaluate_benchmark` | `spec` or a registered name, `n_episodes`, `video=None` |
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

`run_policy` / `eval_policy` / `run_multi_policy` bind the policy's output keys to
the robot's *action* keys via `set_robot_state_keys(robot_action_keys(robot_name))`:
keying by `robot_joint_names` would emit keys that resolve to nothing and leave
those DOFs unmoved (see [Actions](physics.md#actions)). Read the keys a robot
expects with `get_features(robot_name=...)`. A `stop_when` clause, and a benchmark
spec's clauses, are [predicates](predicates.md). Every knob below is documented in
full in the `run_policy` / `eval_policy` docstrings.

## Refusals

Each parameter is checked at the entry point - `start_policy` synchronously, before
the background rollout starts, so a malformed request never returns a false
"started" - together with the provider's own class-level `preflight` hook, and ahead
of the robot claim, so a refused call leaves the robot startable.
`PolicyRunner.run` / `PolicyRunner.evaluate` are drivable directly and raise
`ValueError` there instead, having no envelope to report through.

| Parameter | Domain |
|-----------|--------|
| `n_steps`, `duration`, `control_frequency` | positive. The horizon is `duration` (seconds) or `n_steps` (`duration = n_steps / control_frequency`); `n_steps` wins when both are set, `max_steps` is a legacy alias |
| `action_horizon` | positive whole number - actions consumed from each chunk before it is re-queried. `run_multi_policy` also takes per-robot mappings (`{robot: horizon}`, `instructions={robot: text}`) whose keys must name a robot of that call |
| `n_episodes`, `max_steps` | positive whole number; `max_steps` only when it is the horizon actually read, since a `spec=` call takes its horizon off the benchmark |
| `rtc_inference_timeout_s` | positive finite seconds, or `None` to wait without a deadline |
| `policy_config`, `policy_kwargs` | `dict` - splatted into `create_policy` and into every `get_actions` call |
| `policy_object` | a `Policy` instance, so a provider name or the class is refused by name rather than as an `AttributeError` one layer down |
| `observer`, `on_frame`, `success_fn` | callable, refused before the first step |

The four **posture** flags select a branch rather than scale a quantity -
`fast_mode` (pace the loop at `control_frequency` or run it unpaced),
`reset_between` (reset the scene between episodes or carry the end state over),
`wbc_install_torque_control` (install the WBC torque shim for the call or leave the
actuators alone) and `async_rtc` (overlap inference with actuation or drain each
chunk first) - so a value that is not a boolean is refused rather than read by
truthiness: every non-empty string is truthy, so `fast_mode="false"` would run
unpaced and `async_rtc="false"` would report `rtc_async_enabled=True` beside the
background thread the caller declined, each under `status="success"`. The domain is
the shared `boolean_flag_error` one the recording postures and the mesh wire schema
use, bound to the tool-error envelope through `SimEngine._validate_posture_flags`
and checked ahead of robot resolution, so a refused call builds no policy and
touches no scene. `run_policy` checks all four and treats `async_rtc=None` as its
"resolve from the policy" spelling; `eval_policy` declares `async_rtc` as a plain
`bool` and refuses `None`. Unlike the numeric knobs the check sits at the facades
only: `PolicyRunner.run` takes these flags as the facades hand them and
does not repeat it.

## Stopping, and what counts as running

`stop_policy` is honoured at any point after `start_policy` returns - before the
rollout's first frame, and while it is still queued behind a busy executor. Its
verdict comes from the same in-flight population `list_policies_running` reads, so
the two never report opposite facts about one robot at one instant, and that
population counts either launch shape: a blocking `run_policy` registers no future,
yet is reported as running, is named, and is halted by a stop carrying no
`robot_name` (the shape the mesh e-stop fanout broadcasts). `Was not running on
'<robot>'` is reserved for the genuinely idle case; a stop that finds its own robot
idle while another is mid-rollout names that rollout and the call that ends it, and
carries the reason when that robot's last rollout ended in error.

`list_policies_running` answers on every backend from that population - a MuJoCo,
Newton or Isaac engine names the robots it is driving, and a peer polled over the
mesh reports the same names at the same instant - and `describe()["methods"]` names
it beside `stop_policy`. A backend that can report no population at all is refused
rather than reported as idle: "no policies running" is an affirmative claim about
every robot in the world.

Scene mutations read the same population: `add_robot`, `remove_robot`, `add_object`,
`remove_object`, `move_object`, `add_camera`, `remove_camera`, `load_scene`,
`set_gravity`, `set_timestep` and `reset` refuse while a rollout is driving, naming
the robots in flight and the `stop_policy` remedy, because swapping the compiled
model under a live rollout segfaults. The rollout's own driving thread is exempt, so
a multi-episode rollout still resets between its own episodes.

## What the result reports

`run_policy` returns a `{"json": {...}}` block beside the `text`, mirroring
`eval_policy`:

| Field | Meaning |
|-------|---------|
| `robot_name`, `policy`, `instruction`, `n_steps` | what ran |
| `elapsed_s` | measured on a monotonic clock, so no date correction can move it |
| `stopped_early`, `stopped_reason` | `budget`, `predicate`, `stopped`, or an error |
| `actions_applied`, `steps_advanced` | actions that **commanded** the robot, and physics steps taken - an advanced step is not a commanded action, and an action dict naming no actuator commands nothing |
| `action_errors`, `action_resolution_rate`, `partial_action_failure_rate` | per-key resolution health |
| `video_path` (`None` when no MP4 was written), `video_frames`, `video_fps` | `video_fps` is the rate the MP4 *plays* at - the requested `fps` capped to `control_frequency`, since a rollout renders at most one frame per control step |
| `sim_time_s` | when the backend reports it |
| `stop_when_true_at_reset`, `stop_when_reset_warning` | see [predicates](predicates.md) |
| the `chunk_prefetch_*` fields | see below |

`status` reflects whether the robot *moved*: a run where no step resolved any key,
or named one at all (`actions_applied: 0` - how a model declaring no `<actuator>`
block behaves until `actuate_robot` adds a position servo per joint), returns
`status="error"`, while a run where only *some* keys resolve is operational and adds
an `N/M action steps had unresolved keys` note. Both eval routes tolerate one empty
chunk per step - a policy may legitimately stall - and refuse the *aggregate* when
`actions_applied` is zero, since `success_rate` / `avg_reward` / `pass_hat_k` would
then describe the scene's initial state rather than the policy. A criterion that
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

- [Simulation overview](overview.md) - the scene-construction and rendering verbs.
- [Physics and actions](physics.md) - what a rollout writes every control step.
- [Predicates](predicates.md) - `stop_when` and benchmark clauses.
- [Rollout observers](observers.md) - watching a rollout step by step.
- [Policy providers](../policies/overview.md) - what `policy_provider` may name.
