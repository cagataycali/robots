---
description: "Reinforcement learning from a reward: SimEnv over any SimEngine, the PPO, FastSAC and FastTD3 trainers, every RLTrainSpec field, the checkpoint the rl provider reads."
---

# RL training

By the end of this page you have trained a PPO actor on a MuJoCo `SimEnv`, read its checkpoint back, evaluated it, and know every field the three trainers accept.

```python
import tempfile

from strands_robots.simulation import create_simulation
from strands_robots.simulation.predicates import make_predicate
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv, load_deployable_actor, read_checkpoint_meta


def make_env() -> SimEnv:
    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("so101")
    return SimEnv(
        sim,
        actor_obs_keys=["1", "2", "3", "4", "5", "6"],
        reward_terms=[make_predicate("joint_progress", joint="1", target=0.5)],
        success_fn=make_predicate("joint_above", joint="1", value=0.3),
        max_episode_steps=50,
        action_scale=0.15,
    )


spec = RLTrainSpec(env_factory=make_env, output_dir=tempfile.mkdtemp(), total_timesteps=96, rollout_steps=24, learning_rate=3e-4)
trainer = create_trainer("ppo")
result = trainer.train(spec)
print(result.status, sorted(result.metrics))
meta = read_checkpoint_meta(result.checkpoint_dir)
print({k: meta[k] for k in ("provider", "num_actor_obs", "num_actions", "actor_obs_keys", "action_keys", "hidden_dims")})
print(sorted(trainer.evaluate(num_episodes=2)))
actor = load_deployable_actor(result.checkpoint_dir)
print(type(actor).__name__, actor.actor_obs_keys)
```

You should see (a few `[sim] action value ... outside the range` warnings on the gripper are the untrained actor):

```text
success ['entropy', 'iteration', 'iterations_recorded', 'latest_loss', 'latest_step', 'mean_episode_return', 'mean_reward', 'metrics_path', 'surrogate_loss', 'value_loss']
{'provider': 'ppo', 'num_actor_obs': 6, 'num_actions': 6, 'actor_obs_keys': ['1', '2', '3', '4', '5', '6'], 'action_keys': ['1', '2', '3', '4', '5', '6'], 'hidden_dims': [128, 128]}
['episodes_successful_at_reset', 'max_return', 'mean_length', 'mean_return', 'min_return', 'num_episodes', 'returns', 'std_return', 'success_measured', 'success_rate']
DeployableActor ['1', '2', '3', '4', '5', '6']
```

Real runs need `total_timesteps` in the hundreds of thousands. `create_policy("rl", checkpoint_dir=result.checkpoint_dir)` drives a robot with it ([rl](../policies/rl.md)).

```bash
pip install 'strands-robots[rl]'    # torch + gymnasium + [sim-mujoco]
```

## SimEnv

`SimEnv(engine, actor_obs_keys, reward_terms, *, action_dim=None, robot_name=None, critic_obs_keys=None, max_episode_steps=200, action_scale=1.0, n_substeps=5, success_fn=None, reset_fn=None, device="cpu", skip_images=True)` wraps a live `SimEngine` as one environment of `(1, D)` tensors.

- `actor_obs_keys`: ordered scalar keys from `get_observation` (joint names, `.vel` companions, floating-base keys); the order is part of the weights.
- `reward_terms`: `(sim) -> float` callables summed per step. Build them with `make_predicate` from the [predicates](../simulation/predicates-and-rollouts.md).
- `critic_obs_keys`: privileged sim-only keys for an asymmetric critic.
- `action_dim` defaults to `len(engine.robot_action_keys(robot))`, the actuator count, not always the joint count.
- `action_scale` bounds what the actor commands; `0` disconnects it and is refused.
- `n_substeps=5`: a position servo needs several physics steps per target.
- `success_fn` ends an episode as a real terminal; `max_episode_steps` is a truncation, value-bootstrapped by the trainers. Without `success_fn`, `evaluate` reports `success_measured=False` and a `success_rate` of zero measuring nothing.

`VecSimEnv(env_factory, num_envs)` steps N independent `SimEnv` through one thread pool, stacks to `(N, D)`, keeping the terminal observation in `infos[i]["terminal_obs"]` across autoreset. `GymSimEnv(sim_env)` is the `gymnasium.Env` wrapper.

## Trainers

| provider | class | family | own fields |
|---|---|---|---|
| `ppo` | `PpoTrainer` | on-policy, GAE, clipped surrogate | `gamma`, `lam`, `clip_param`, `num_learning_epochs`, `num_mini_batches`, `entropy_coef`, `value_loss_coef`, `max_grad_norm`, `init_noise_std`, `normalize_advantage` |
| `fast_sac` | `FastSacTrainer` | off-policy, replay buffer, entropy temperature | `buffer_size`, `batch_size`, `learning_starts`, `gradient_steps`, `tau`, `autotune_alpha`, `init_alpha`, `alpha_lr`, `target_entropy` |
| `fast_td3` | `FastTd3Trainer` | off-policy, twin critics, delayed actor | the SAC buffer fields plus `policy_delay`, `exploration_noise_std`, `target_noise_std`, `target_noise_clip` |

All three share `setup`, `save_checkpoint`, `load_checkpoint`, `latest_checkpoint` and `evaluate(spec=None, checkpoint_dir=None, num_episodes=10)`; `train` fails closed via `validate`. `evaluate` updates nothing: mean action, no gradients, normalizers frozen.

## RLTrainSpec

Extends `TrainSpec` (so `output_dir`, `learning_rate`, `seed` and the rest are there) with:

| field | default | read by |
|---|---|---|
| `env_factory` | required | all; a zero-arg callable returning a fresh `SimEnv` |
| `total_timesteps` | `100_000` | all |
| `rollout_steps` | `24` | all |
| `num_envs` | `1` | ppo (`>1` wraps `VecSimEnv`); fast_sac and fast_td3 refuse anything but `1` |
| `actor_obs_keys`, `critic_obs_keys` | `[]` (from the env) | all |
| `gamma` | `0.99` | all |
| `lam` | `0.95` | ppo |
| `clip_param` | `0.2` | ppo |
| `num_learning_epochs`, `num_mini_batches` | `5`, `4` | ppo |
| `entropy_coef`, `value_loss_coef`, `max_grad_norm` | `0.0`, `1.0`, `1.0` | ppo |
| `hidden_dims` | `(128, 128)` | all |
| `init_noise_std` | `1.0` | ppo |
| `normalize_obs`, `normalize_advantage` | `True`, `True` | all; ppo |
| `device` | `None` (auto) | all |
| `log_interval` | `10` | all; checkpoint cadence in iterations |
| `buffer_size`, `batch_size`, `learning_starts`, `gradient_steps`, `tau` | `100_000`, `256`, `1_000`, `1`, `0.005` | fast_sac, fast_td3 |
| `autotune_alpha`, `init_alpha`, `alpha_lr`, `target_entropy` | `True`, `1.0`, `3e-4`, `None` | fast_sac |
| `policy_delay`, `exploration_noise_std`, `target_noise_std`, `target_noise_clip` | `2`, `0.1`, `0.2`, `0.5` | fast_td3 |
| `learning_rate` | `1e-4` | all |

Booleans are checked, not read by truthiness; counts are positive integers; the two loss coefficients accept any finite real.

## The checkpoint

`save_checkpoint` writes `policy.pt` (the `state_dict`, the frozen `EmpiricalNormalization`, `provider`) and `policy_meta.json` with `provider`, `num_actor_obs`, `num_critic_obs`, `num_actions`, `actor_obs_keys`, `action_keys`, `hidden_dims`, `iteration`. `read_checkpoint_meta` refuses a file missing any of the first six, by name. `load_deployable_actor(checkpoint_dir, device)` rebuilds the network the `provider` names (PPO raw means, FastTD3 a `tanh`, FastSAC a squashed mean/log-std pair), restores weights and normalizer, and returns the `DeployableActor` whose `act(obs)` `create_policy("rl")` calls.

## Limits

- CPU MuJoCo is the only in-process batched path (`VecSimEnv` threads N engines); GPU-parallel RL: [isaaclab](isaaclab.md).
- No image observations: `actor_obs_keys` are scalars, `skip_images=True` by default.
- Three algorithms, one MLP shape each, no recurrent actor; curriculum is whatever `reset_fn` and the terrain `difficulty` knob give.
