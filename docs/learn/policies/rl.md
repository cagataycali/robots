---
description: rl rolls out an actor trained by create_trainer("ppo" | "fast_sac" | "fast_td3") from its policy.pt and policy_meta.json.
---

# rl

By the end of this page you can load an actor the RL trainers wrote and drive a robot with it through the same `run_policy` path as every other provider.

```bash
pip install 'strands-robots[rl]'    # torch + gymnasium + the MuJoCo backend
```

## What it is

`RLCheckpointPolicy` is the inference half of the RL loop. `create_trainer("ppo" | "fast_sac" | "fast_td3")` trains against a `SimEnv` and writes `policy.pt` plus `policy_meta.json`; this provider loads that pair and presents it as an ordinary `Policy`. Rollout is deterministic: the actor's mean action, no sampling. `requires_images` is `False`.

The checkpoint's `actor_obs_keys` are read from the observation by name, in the trained order, because the order is part of the weights. A key the observation does not carry is refused rather than defaulted; a zero would command a real robot from a fabricated state.

```python
from strands_robots.policies import create_policy

try:
    create_policy("rl")
except ValueError as exc:
    print(exc)
```

You should see:

```text
checkpoint_dir is required for the 'rl' policy provider: pass the directory a trainer wrote (TrainResult.checkpoint_dir), e.g. create_policy('rl', checkpoint_dir=result.checkpoint_dir)
```

## Constructor keywords

{{providers:kwargs:rl}}

`checkpoint_dir` is spelled as the trainers spell it (`TrainResult.checkpoint_dir`, `latest_checkpoint`), so the value you already hold is the value this takes. `device` defaults to `cpu`; PPO on MuJoCo declares no GPU floor.

## Train, then roll out

```python
import os
import tempfile

from strands_robots.simulation import create_simulation
from strands_robots.simulation.predicates import _joint_progress
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv


def make_env() -> SimEnv:
    sim = create_simulation("mujoco", mesh=False)
    sim.create_world()
    sim.add_robot("so101")
    return SimEnv(sim, actor_obs_keys=["1", "2", "3", "4", "5", "6"], reward_terms=[_joint_progress("1", 0.5)], max_episode_steps=50, action_scale=0.15)


spec = RLTrainSpec(env_factory=make_env, output_dir=tempfile.mkdtemp(), total_timesteps=96, rollout_steps=24, learning_rate=3e-4)
result = create_trainer("ppo").train(spec)
print(result.status, sorted(os.listdir(result.checkpoint_dir)))

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
out = sim.run_policy(robot_name="so101", policy_provider="rl", policy_config={"checkpoint_dir": result.checkpoint_dir}, n_steps=20, control_frequency=50.0)
print(out["status"])
sim.cleanup()
```

You should see (the numbers on your machine differ, the files do not):

```text
success ['policy.pt', 'policy_meta.json']
success
```

A few `[sim] action value ... outside the range` warnings on the gripper are expected: the actor is untrained. Six optimizer iterations train nothing useful; the point is that the checkpoint a trainer writes is the value `rl` takes. Real runs use `total_timesteps` in the hundreds of thousands.

The trainers, their fields and what `policy_meta.json` records are on the [RL training](../training/rl.md) page.

## rsl_rl_onnx

`policy="rsl_rl_onnx"` loads an actor exported by the [mjlab](../simulation/mjlab.md) trainer (`train_policy(provider="rsl_rl")`): `onnx_path` (local or `hf://repo/file.onnx`) and `robot`. The ONNX metadata carries joint names, default pose, action scale and the observation terms, so the same file runs on `mujoco`, `mjlab` and hardware; velocity tasks take `target_velocity`, reach tasks `target_pose`.

## Limits

- The observation must carry every `actor_obs_keys` name. Rolling a checkpoint out on a robot whose joints are named differently is refused, not remapped.
- Deterministic mean action only. There is no exploration noise at inference.
- Only checkpoints written by this package's trainers load; a foreign `policy.pt` has no `policy_meta.json` and is refused by `FileNotFoundError`.
