---
description: The read-only observer lane that reports every applied action of a rollout.
---

# Rollout observers

## Watching a rollout: the `observer` lane

`run_policy(observer=...)` takes a read-only callable receiving one
`RunPolicyStarted`, one `RunPolicyStep` per completed `send_action`, and one
`RunPolicyEnded`. It is a *second* lane beside the backend's `on_frame` hook rather
than a use of it: that hook carries cooperative cancellation, the trajectory mirror,
mesh step telemetry and dataset recording, so supplying one would remove all of that
instead of adding observation.

```python
from strands_robots.simulation.observers import RunPolicyStep

def watch(event):
    if isinstance(event, RunPolicyStep) and event.action_resolution != "full":
        print(event.applied_action_index, event.unresolved_action_keys)

sim.run_policy(robot_name="alice", policy_provider="mock", observer=watch)
```

The events use observer schema version **2** and carry four things `on_frame`'s
`(step, obs, action)` signature cannot:

| Field | Meaning |
|-------|---------|
| `action_resolution` | The backend's per-key `send_action` verdict as `full` / `partial` / `none` / `unknown`. `partial` and `none` need a complete per-key breakdown; a coarse backend error is `unknown` with empty key tuples, since input keys are not proof of what reached physical state - such steps stay in `action_errors` but leave the aggregate rate denominators alone |
| `observation_is_chunk_reused` | `true` only for a later action reusing the chunk-start snapshot. Not authoritative freshness: the first action after a prefetch swap can already use an old snapshot |
| `observation_age_steps` | Authoritative nonnegative age in control steps. A sync chunk reports its chunk index; an async chunk carries the prefetch sample's remaining old-chunk attempts across the swap and adds the new index. An active recording refreshes every step and reports `0` |
| `legacy_hook_outcome` | What the backend's hook did: `ok`, `cancelled`, `recording_error`, `error` or `absent` |

`applied_action_index == legacy_step_index` for every step, including a cancelled or
recording-failed one - the abort shows in `legacy_hook_outcome`, not as an index
offset. `step_count` increments after the hook, so `RunPolicyEnded.applied_actions`
can exceed `legacy_steps_used` by one when that final hook aborts. `event_seq` is
dense and 0-based within one `run_id` so a gap is observable, `monotonic_ns` orders
the stream, and `utc_ns` is derived from a single rollout anchor. Each episode of a
multi-episode call is its own lifecycle with its own `run_id`; once
`RunPolicyStarted` dispatch is attempted, exactly one `RunPolicyEnded` dispatch is
attempted on every Python exit, and a preflight refusal opens no lifecycle.

Five rules the lane holds to:

- **Additive.** It changes no applied action and no existing result-json field, and
  adds one key, `observer_failures`.
- **Contained.** An `Exception` never alters the rollout outcome and never reaches
  the `max_onframe_failures` watchdog, which exists for a recorder losing dataset
  frames. `CooperativeStop` is contained by name - it is a `BaseException` so that a
  broad `except Exception` cannot swallow a cancellation, and unnamed here any
  observer could cancel a rollout it is only meant to watch. Contained failures are
  counted in `observer_failures`, so a stream with holes says so.
- **Except the four that are nobody's telemetry.** `KeyboardInterrupt`,
  `SystemExit`, `GeneratorExit` and `asyncio.CancelledError` propagate: none is an
  `Exception` subclass, and a generator closed underneath a visualiser is a teardown
  rather than a drawing failure. Raised while another exception unwinds, the original
  stays primary and this one is attached as a note.
- **Borrowed, not copied.** `observation` and `action` are the objects the hook
  received: read-only, and not to be retained past the call.
- **Not isolated.** Dispatch is synchronous on the rollout thread, so a blocking
  observer blocks the robot. The loop paces on a *deadline*, so a consumer has one
  control period (`1 / control_frequency`) for free; overrun it and the loop drops
  the missed deadline rather than firing catch-up actions, so the arm sees a gap.

Scope: `run_policy` (including `n_episodes > 1`) and `PolicyRunner.run`.
`eval_policy`, `evaluate_benchmark` and `run_multi_policy` are separate loops with
different step semantics and carry no observer yet.

## See also

- [Policy rollouts](rollouts.md) - the `run_policy` surface this lane watches.
