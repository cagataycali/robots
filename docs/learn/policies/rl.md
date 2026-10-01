---
description: rl rolls out an actor trained by create_trainer("ppo" | "fast_sac" | "fast_td3"), or an rsl_rl run from Isaac Lab, from disk or the Hub.
---

# rl

By the end of this page you can load an actor the RL trainers or Isaac Lab wrote and drive a robot through the same `run_policy` path as every other provider.

```bash
pip install 'strands-robots[rl]'    # torch + gymnasium + MuJoCo
```

## What it is

`RLCheckpointPolicy` is the inference half of the RL loop. `create_trainer("ppo" | "fast_sac" | "fast_td3")` trains against a `SimEnv` and writes `policy.pt` plus `policy_meta.json`; this provider presents that pair as an ordinary `Policy`. Rollout is the actor's mean action, no sampling. `requires_images` is `False`.

The checkpoint's `actor_obs_keys` are read from the observation by name, in the trained order (the order is part of the weights); a missing key is refused, never defaulted to a zero that would command a real robot from a fabricated state.

## Three shapes of `checkpoint_dir`

```python
from strands_robots.policies import create_policy

create_policy("rl", checkpoint_dir=result.checkpoint_dir)  # strands checkpoint dir
create_policy("rl", checkpoint_dir="logs/rsl_rl/h1_rough")  # rsl_rl run, or one model_<n>.pt
create_policy("rl", checkpoint_dir="owner/name@main")  # Hub repo id; @revision optional
```

An rsl_rl actor (what `isaaclab` trains: ELU layers plus observation normalizer) is rebuilt once into `<run>/strands_policy/`, reused while newer than the model. `action_names` in a `record.json` beside it name the actions when the count matches. From the Hub only `model_*.pt`, `params/agent.yaml`, `record.json` and the strands pair are fetched; a repo that cannot be downloaded is a `RuntimeError` naming the id.

## Constructor keywords

{{providers:kwargs:rl}}

`device` defaults to `cpu`: PPO on MuJoCo declares no GPU floor.

## Train, then roll out

```python
import tempfile

from strands_robots.simulation import create_simulation
from strands_robots.simulation.predicates import _joint_progress
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv


def make_env() -> SimEnv:
    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("so101")
    return SimEnv(sim, actor_obs_keys=["1", "2", "3", "4", "5", "6"], reward_terms=[_joint_progress("1", 0.5)], max_episode_steps=50, action_scale=0.15)


spec = RLTrainSpec(env_factory=make_env, output_dir=tempfile.mkdtemp(), total_timesteps=96, rollout_steps=24, learning_rate=3e-4)
result = create_trainer("ppo").train(spec)
print(result.status)

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
out = sim.run_policy(robot_name="so101", policy_provider="rl", policy_config={"checkpoint_dir": result.checkpoint_dir}, n_steps=20, control_frequency=50.0)
print(out["status"])
sim.cleanup()
```

You should see `success` twice, plus a few `[sim] action value ... outside the range` gripper warnings: six optimizer iterations train nothing; the point is that a trainer's checkpoint is what `rl` takes.

Trainers and `policy_meta.json` fields: [RL training](../training/rl.md). Isaac Lab runs: [Isaac Lab training](../training/isaaclab.md).

## rsl_rl_onnx

`policy="rsl_rl_onnx"` loads an actor exported by the [mjlab](../simulation/mjlab.md) trainer (`train_policy(provider="rsl_rl")`): `onnx_path` (local or `hf://repo/file.onnx`) and `robot`. The ONNX metadata carries joint names, default pose, action scale and observation terms, so one file runs on `mujoco`, `mjlab` and hardware; velocity tasks take `target_velocity`, reach tasks `target_pose`.

## Limits

- The observation must carry every `actor_obs_keys` name; differently named joints are refused, not remapped.
- Deterministic mean action only, no exploration noise.
- Only strands pairs and rsl_rl 5.x actors (`actor_state_dict`, `mlp.<i>` layers) load; anything else is a `FileNotFoundError`.
