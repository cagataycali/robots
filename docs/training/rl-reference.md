---
description: RL API reference - SimEnv argument domains, the BaseRLAlgo lifecycle and evaluate(), the device rule, and every RLTrainSpec field validate() grades.
---

# RL API reference

The domains and contracts behind [Reinforcement learning (from
scratch)](rl.md). Everything here is graded before a run spends compute:
`SimEnv` checks its arguments at construction, and `Trainer.validate(spec)`
reports problems before `setup()` builds anything.

## SimEnv numeric arguments

An unusable value raises `ValueError` naming the class and the argument
(`SimEnv: action_scale must be > 0, got 0.0.`):

| Argument | Domain | Why |
|---|---|---|
| `action_scale` | positive finite | Multiplies every action sent. `0` disconnects the policy from the robot, a negative inverts every DOF, `nan`/`inf` make each command unsendable. |
| `max_episode_steps` | positive whole number | `0` or below times out on the first step, as a *truncation* that GAE value-bootstraps. |
| `n_substeps` | positive whole number | A position target the PD controller needs several substeps to track. |
| `action_dim` | positive integer, or `None` | `None` sizes the head from the robot's action keys; `0` gives the policy no outputs. |

The command is clamped to each actuator's `ctrlrange`, so a `tanh`-squashed actor
(`[-1, 1]`) reaches only the part overlapping `[-action_scale, action_scale]`. At
the default `1.0` that is 46.4% of the so100's ranges, 57.6% of the g1's and 3.5%
of the go2's torque limits; asymmetric ranges need a per-actuator mapping.

Nothing is refused that the code downstream accepts: `0.25`,
`np.float32(0.25)`, `50.0` and `np.int64(50)` all normalize to the type
the attribute advertises.

## BaseRLAlgo

`BaseRLAlgo` is the abstract RL trainer - a `Trainer` subclass, so RL flows
through the same `create_trainer` / `validate` / `export` contract while adding
`setup`, `collect_rollout`, `update` and `save_checkpoint`. The default
`train()` runs the on-policy loop; off-policy trainers override it with a
replay-buffer loop, keeping the same hooks and checkpoint format.

`evaluate(spec=None, checkpoint_dir=None, num_episodes=10)` is the eval peer of
`train()`: deterministic (mean) action, gradients disabled, normalization
frozen. Returns `num_episodes`, `mean_return`, `std_return`, `min_return`,
`max_return`, `mean_length`, per-episode `returns`, `success_rate` (fraction
ending on a genuine `success_fn` terminal, not a time-out), and two fields:

| Field | Value | What it means | Rate it forces |
| --- | --- | --- | --- |
| `success_measured` | `False` | no `success_fn`, every episode times out | hard `0.0` |
| `episodes_successful_at_reset` | `> 0` | predicate held at reset, episodes terminate on step one | hard `1.0` each |

Both are warnings. A hard `0.0` is indistinguishable from a policy that failed
everything; a hard `1.0` is what commanding the current pose earns. Neither
changes a returned figure. `PolicyRunner.evaluate` and `evaluate_benchmark`
report the same two facts. `evaluate()` restores train/eval mode on **every**
exit, including a raising one.

## RLTrainSpec

`RLTrainSpec` extends `TrainSpec`, ignores dataset fields and reads
`env_factory`, `total_timesteps`, `rollout_steps`, `num_envs`, the PPO
hyperparameters, the [off-policy SAC fields](rl.md#fastsac), the [TD3
fields](rl.md#fasttd3), plus `output_dir` / `learning_rate` / `seed` / `device`.

`validate()` grades every field below before `setup()` builds anything, and
*reports* problems rather than raising. A value outside a domain gives a
differently-shaped run torch honors silently under `status="success"`.

| Field | Domain | Graded by | An unusable value |
|---|---|---|---|
| `total_timesteps`, `rollout_steps` | positive integer | all three | Factors of `num_iters`: a fraction/`nan`/`inf` clamps the run to one iteration; `rollout_steps=True` normalizes over a length-one batch. |
| `num_envs` | PPO/FastTD3 `>= 1`, FastSAC exactly `1` | each backend | The constraint is the trainer's, not MuJoCo's. |
| `learning_starts` | positive integer, `>= batch_size`, reachable by step budget and `buffer_size` | off-policy | Short of either bound: **zero** gradient steps, exports the initialized network. |
| `hidden_dims` | sequence of positive integer widths; empty = linear | all three | `nn.Linear(0, ...)` emits its bias alone - one fixed action in every state. |
| `gamma` | finite, `[0, 1]` | all three | Above 1 the return diverges. `1` = undiscounted, `0` = myopic. |
| `tau` | finite, `(0, 1]` | off-policy | Polyak coefficient of the target critics. |
| `batch_size`, `buffer_size`, `gradient_steps` | positive integer | off-policy | Sample size, capacity, updates per iteration. |
| `num_learning_epochs` | positive integer | PPO | Non-positive: no gradient step, losses `0.0`. |
| `clip_param` | positive; `inf` = no clip | PPO | `nan` silently removes the trust region; `0` inverts the clamp. |
| `max_grad_norm` | positive, 64-bit; `inf` = no clip | PPO | `0` zeroes every gradient; negative = gradient ascent. |
| `init_alpha` | positive finite | FastSAC | `0` -> `log(0) = -inf`, entropy term dropped. |
| `alpha_lr` | positive finite, when `autotune_alpha` | FastSAC | `0` freezes temperature; `inf` sends it to infinity at once. |
| `target_entropy` | finite real or `None` | FastSAC | `None` uses the `-num_actions` heuristic. |
| `policy_delay` | positive integer | FastTD3 | Modulus of the actor-update gate: `0` divides by zero, `True` = no delay. |
| `exploration_noise_std` | positive finite | FastTD3 | `0` removes exploration after the warmup. |
| `target_noise_std`, `target_noise_clip` | positive finite | FastTD3 | Either at `0` removes target-policy smoothing. |
| `normalize_obs`, `normalize_advantage`, `autotune_alpha` | `bool` | its backend | Truthiness: `"false"` selects the affirmative branch. |
| `log_interval` | whole number of iterations | all three | **Checkpoint cadence**: `0` = final only; `nan` is silently that mode. |
| `device` | torch device string | all three | Spelling only - see [Device selection](#device-selection). |

`evaluate()`'s `num_episodes` takes the same positive-integer domain.

## Device selection

The learner goes on `RLTrainSpec.device`, defaulting to `cuda` when available.
`setup()` reconciles the `SimEnv` onto it. Pass `device="cpu"` to stay on CPU.
`validate()` grades only the *spelling* - `device="cuda"` on a CPU-only host is
valid, because a queued run may execute elsewhere. All three trainers train fine
on CPU: MuJoCo stepping dominates.

## See also

- [Reinforcement learning (from scratch)](rl.md) - the trainers, the worked
  example and the rollout artifacts.
- [Training overview](overview.md) - `TrainSpec` and the supervised backends.
