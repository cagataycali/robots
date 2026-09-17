---
description: The predicate DSL behind stop_when, benchmark clauses and dense rewards.
---

# Predicates

One clause language compiles `run_policy(stop_when=...)`, a benchmark spec's
`success` / `failure` / `dense_reward`, and the stages of a `staged_reward`. Every
clause passes through `make_predicate`, which holds its kwargs to a domain at
authoring time rather than one rollout later:

| Kwarg | Domain | Unchecked, the clause |
|-------|--------|-----------------------|
| numeric | finite | `nan` compiles clean and makes every comparison `False`, so it never fires and the rollout spends its whole budget on an honest miss |
| a tolerance - `tol`, `threshold`, any `*_tol` | `>= 0` | reads as a *wider* bound, while it bounds a distance or a squared magnitude that is never negative |
| a signed value - `body_on`'s `z_offset`, `base_velocity`'s `vx`, `body_below_z`'s `z` | both signs kept | the sign is part of the value |
| an entity name - a body, joint, geom, container, or `grasped`'s `gripper_prefix` | non-empty string | can report a success that never happened: `gripper_prefix` is matched with `startswith`, so a blank one selects **every** geom and `{predicate: grasped, body: cube, gripper_prefix: ""}` fires on the cube's contact with the floor it was placed on, scoring `success_rate: 1.0` under `success_measured: true` |

The `base_*` family keeps `robot=None`, its documented "the sole robot" spelling.

## Names are probed against the live scene

Before the rollout starts, every entity a clause references is resolved through the
same lookup the predicate uses at evaluation time: bodies via `get_body_state`,
joints via `get_observation`, the `base_*` family via the `base_pos` / `base_quat`
signals. So `body: cubeee` is a structured error rather than a rollout reporting
`stopped_reason="budget"`, or an `evaluate_benchmark` reporting `success_rate: 0.0`
beside `success_measured: true` - indistinguishable from an honest policy failure.
The base family is collected by *predicate* rather than by kwarg, because its
`robot` defaults; that also catches arming a base term on a robot with no floating
base, which reports neither signal (`base_tipped` on an SO-100 is permanently
`False`). Geom names (`contact_between`) are uncollected: there is no generic geom
lookup on the engine ABC.

Two shapes report no entities and are evaluated unchanged: a spec declaring its own
`scene`, whose bodies `on_episode_start` creates after the probe would run, and a
`DeclarativeBenchmark` built from already-compiled callables. A benchmark written in
Python opts into the probe by exposing `referenced_entities()`.

## A clause the scene already satisfies

No compile-time check can see this - it is a fact about the initial state, and
domain randomisation legitimately draws one on some episodes - so it is reported
rather than refused. Both eval routes and `stop_when` sample only after an applied
action, so such a clause fires on the first step whatever the policy commands:

| Surface | Reports |
|---------|---------|
| `run_policy` | `stop_when_true_at_reset` (bool, always present) with `stop_when_reset_warning` beside it |
| `evaluate_benchmark` | `episodes_failed_at_reset` and `reset_failure_warning`, plus `failure_at_reset` per episode |
| `eval_policy` | neither - it takes a `success_fn` and has no failure criterion to sample |

Every other figure is left as measured.

## Nesting

`staged_reward` is itself a registered predicate, so a stage's `reward` may be
another `staged_reward` and a curriculum can be a machine of machines. Each machine
clears its own sub-terms on reset - by the same rule, anything exposing a zero-arg
`reset()` - or a second episode opens inside a sub-curriculum it has not earned,
never emitting its shaping signal again and paying its one-time `bonus` once per
process instead of once per episode.

## See also

- [Policy rollouts](rollouts.md) - `stop_when` and the eval routes that read these clauses.
- [Reinforcement learning](../training/rl.md) - rewards built from the same terms.
