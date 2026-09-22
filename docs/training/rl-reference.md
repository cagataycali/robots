---
description: RL API reference - SimEnv argument domains, the BaseRLAlgo lifecycle and evaluate(), the device rule, and every RLTrainSpec field validate() grades.
---

# RL API reference

The domains and contracts behind [Reinforcement learning (from
scratch)](rl.md). Everything here is graded before a run spends compute:
`SimEnv` checks its arguments at construction, and `Trainer.validate(spec)`
reports problems before `setup()` builds an environment, a network or an
optimizer.

## SimEnv numeric arguments

Each is checked at construction, before the engine is read; an unusable value
raises `ValueError` naming the class and the argument
(`SimEnv: action_scale must be > 0, got 0.0.`):

| Argument | Domain | Why |
|---|---|---|
| `action_scale` | positive finite | It multiplies every action sent: `0` disconnects the policy from the robot, a negative inverts every DOF, and `nan`/`inf` make each command unsendable - the rollout then banks its full return having moved nothing. |
| `max_episode_steps` | positive whole number | `0` or below times out on the first step, as a *truncation* - which on-policy GAE value-bootstraps. |
| `n_substeps` | positive whole number | The action is a position target; the PD controller needs several substeps to track it. `send_action`'s own domain. |
| `action_dim` | positive integer, or `None` | `None` sizes the head from the robot's action keys; a width of `0` gives the policy no outputs. |

Nothing is refused that the code downstream accepts: `0.25`,
`np.float32(0.25)`, `50.0` and `np.int64(50)` all normalize to the `float`/`int`
the attribute advertises.

## BaseRLAlgo

`BaseRLAlgo` is the abstract RL trainer - a `Trainer` subclass, so RL flows
through the same `create_trainer` / `validate` / `export` contract while adding
the hooks `setup`, `collect_rollout`, `update` and `save_checkpoint`. The
default `train()` runs the on-policy loop over them; an off-policy algorithm
overrides `train()` with a replay-buffer loop, keeping the same hooks and
checkpoint format.

`evaluate(spec=None, checkpoint_dir=None, num_episodes=10)` is the eval peer of
`train()`: the deterministic (mean) action, gradients disabled, normalization
frozen, so its numbers are what a deployed `policy.pt` produces. It returns
`num_episodes`, `mean_return`, `std_return`, `min_return`, `max_return`,
`mean_length`, the per-episode `returns`, `success_rate` (the fraction of
episodes that ended on a genuine terminal via the env's `success_fn`, not a
time-out), and two fields that say whether that rate measured the policy:

| Field | Value | What it means | Rate it forces |
| --- | --- | --- | --- |
| `success_measured` | `False` | the env has no `success_fn`, so nothing can terminate and every episode times out | hard `0.0` |
| `episodes_successful_at_reset` | `> 0` | the predicate already held at reset, so those episodes terminate on their first step whatever the policy commands | hard `1.0` each |

Both are logged as warnings. Read them before trusting `success_rate`: a hard
`0.0` is indistinguishable from a policy that was scored and failed everything,
and a hard `1.0` is what a policy commanding its own current pose earns - so
usually a threshold on the wrong side of the initial state. Neither field
changes a returned figure, because `reset_fn` legitimately draws a new initial
state per episode. `PolicyRunner.evaluate` and `evaluate_benchmark` report the
same two facts. `evaluate()` restores train/eval mode on **every** exit,
including a raising one, so `train -> evaluate -> train` resumes with the
running statistics still learning.

## RLTrainSpec

`RLTrainSpec` extends `TrainSpec`. RL ignores the dataset fields
(`dataset_root` etc.) and reads `env_factory`, `total_timesteps`,
`rollout_steps`, `num_envs`, the PPO hyperparameters (`gamma`, `lam`,
`clip_param`, `num_learning_epochs`, `num_mini_batches`, `entropy_coef`,
`value_loss_coef`, `max_grad_norm`, `hidden_dims`, `init_noise_std`), the
[off-policy SAC fields](rl.md#fastsac), plus the universal `output_dir` /
`learning_rate` / `seed` / `device`.

`validate()` grades the fields below before `setup()` builds an environment, a
network or an optimizer, and *reports* every problem rather than raising. Each
domain is the one its consumer can honor: a value outside it does not give a
slower run but a differently-shaped one that torch honors silently, under
`status="success"` and a written checkpoint - so each row names what it does.

| Field | Domain | Graded by | An unusable value |
|---|---|---|---|
| `total_timesteps`, `rollout_steps` | positive integer | both | The factors of `num_iters = max(1, total_timesteps // (rollout_steps * num_envs))`: a fraction, `nan` or `inf` clamps the run to one iteration, and `rollout_steps=True` makes PPO normalize advantages over a length-one batch. |
| `num_envs` | PPO `>= 1`, FastSAC exactly `1` | each backend | PPO parallelizes; the MuJoCo-backed FastSAC is single-env. |
| `learning_starts` | positive integer, `>= batch_size`, and reachable by both the collected step budget and `buffer_size` | off-policy | Short of either bound the run takes **zero** gradient steps and exports the network `setup` initialized. Each short count is reported on its own. |
| `hidden_dims` | sequence of positive integer widths; empty means a linear policy | all three | `nn.Linear` accepts a width of zero, and the layer after it then emits its bias alone - so the actor commands one fixed action in every state. A problem names the index (`hidden_dims[1] must be a positive integer`). |
| `gamma` | finite, `[0, 1]` | both | The discounted return is a geometric series, so above 1 it diverges over the horizon. Both endpoints are in: `1` undiscounted, `0` myopic. |
| `tau` | finite, `(0, 1]` | FastSAC | The Polyak coefficient of the target critics. |
| `num_learning_epochs` | positive integer | PPO | The bound of the whole optimizer loop: non-positive takes no gradient step and reports losses of `0.0`. |
| `clip_param` | positive; `inf` means do not clip | PPO | The half-width of the trust region. `nan` silently removes it - comparisons against `nan` are false, so the gradient takes the unclipped branch - and `0` or a negative inverts the clamp into a constant. |
| `max_grad_norm` | positive and 64-bit-representable; `inf` means do not clip | PPO | `clip_grad_norm_` scales by `max_norm / total_norm` without judging it: `0` zeroes every gradient, a negative makes the update gradient *ascent*, and `10**400` is refused on range rather than sign. |
| `init_alpha` | positive finite | FastSAC | The temperature is stored as its logarithm, so `0` gives `-inf` and drops the entropy term from both losses for good. Read whether or not tuning is on. |
| `alpha_lr` | positive finite, when `autotune_alpha` | FastSAC | `0` never moves the temperature, so the tuning asked for does not happen; `inf` sends it to an infinity on the first step. |
| `target_entropy` | finite real of **either sign**, or `None` for the `-num_actions` heuristic | FastSAC | Its default is negative, so no endpoint is decidable. |
| `normalize_obs`, `normalize_advantage`, `autotune_alpha` | `bool` | the backend that reads it | Each is spent by truthiness, so `"false"`, `"no"` and `"0"` would select the affirmative branch. `autotune_alpha` is graded ahead of the `alpha_lr` check it gates. |
| `log_interval` | whole number of iterations | all three | **The checkpoint cadence, not a logging one**: `it % log_interval == 0` decides whether `save_checkpoint` runs, so it is RL's [`TrainSpec.save_freq`](overview.md). `0` is the supported "final checkpoint only" mode, and `nan` passes the truthiness guard but never the modulus, so it is silently that mode - costly, because RL return is non-monotonic and the deployable policy is often an earlier iteration. |
| `device` | a device string torch can parse | all three | Spelling only, on the same domain `lerobot_train` and `LerobotTrainer` apply - see [Device selection](#device-selection). |

`evaluate()`'s `num_episodes` takes the same count domain: it is the `range()`
bound of the episode loop and the denominator of `success_rate`.

## Device selection


The learner (actor-critic, normalizers, rollout buffers) goes on
`RLTrainSpec.device`, defaulting to `cuda` when available, and that device is
authoritative: `setup()` reconciles the `SimEnv` onto it, so observations,
rewards and dones are built where the network is. Pass `device="cpu"` to stay on
CPU on a GPU host. `validate()` grades only the *spelling* - `device="cuda"` on
a CPU-only host is valid, because a queued run is written on one machine and
executed on another. Both trainers train fine on CPU: MuJoCo stepping dominates.

## See also

- [Reinforcement learning (from scratch)](rl.md) - the trainers, the worked
  example and the rollout artifacts.
- [Training overview](overview.md) - `TrainSpec` and the supervised backends.
