---
description: What trains where: the Trainer contract, the nine trainers create_trainer knows, TrainSpec, the validate to export lifecycle, the two agent tools.
---

# Training

By the end of this page you can name every trainer, what it drives on what hardware, and run the `validate` to `export` lifecycle with the mock trainer.

```python
import json
import os
import tempfile

from strands_robots.training import TrainSpec, create_trainer, list_trainers

print(list_trainers())
root = tempfile.mkdtemp()
os.makedirs(f"{root}/meta")
with open(f"{root}/meta/info.json", "w") as fh:
    json.dump({"codebase_version": "v3.0", "total_episodes": 10, "total_frames": 3000, "fps": 30}, fh)

trainer = create_trainer("mock")
spec = TrainSpec(dataset_root=root, base_model="lerobot/smolvla_base", output_dir=tempfile.mkdtemp(), steps=10)
print(trainer.validate(spec))
result = trainer.train(spec)
print(result.status, sorted(os.listdir(result.checkpoint_dir)), sorted(result.metrics))
print(trainer.status(result.job_id).status, trainer.export(spec, result.checkpoint_dir) is not None)
print({name: create_trainer(name).hardware_floor for name in ("lerobot_local", "cosmos3")})
```

You should see:

```text
['cosmos3', 'fast_sac', 'fast_td3', 'isaaclab', 'lerobot_local', 'mock', 'ppo', 'sagemaker']
[]
success ['config.json'] ['latest_loss', 'latest_step', 'learning', 'liveness_ok']
success True
{'lerobot_local': {'min_gpus': 1, 'min_vram_gb': 8, 'multinode': False}, 'cosmos3': {'min_gpus': 8, 'min_vram_gb': 80, 'multinode': True}}
```

## What trains where

One name owns both halves: `create_policy("lerobot_local")` runs what `create_trainer("lerobot_local")` wrote, and `create_policy("rl", checkpoint_dir=...)` runs the RL trainers' `policy.pt`. Supervised trainers are declared in `registry/policies.json` under a `trainer` block; the rest register through `register_trainer`, as yours can.

| trainer | drives | in process | floor |
|---|---|---|---|
| `lerobot_local` | `lerobot.scripts.lerobot_train.train(cfg)` for any LeRobot policy (`act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `groot`, ...) or reward model (`sarm`) | yes, no subprocess | 1 GPU, 8 GB |
| `cosmos3` | `cosmos_framework` SFT: `prepare()` converts the HF checkpoint to DCP, `train()` calls `scripts.train.launch` with a TOML recipe plus overrides | yes | 8 GPUs, 80 GB, multinode |
| `sagemaker` | submits the same `TrainSpec` to one managed SageMaker job in a caller-supplied `image_uri`; reimplements nothing | no, AWS | `instance_type="ml.g5.xlarge"` default |
| `isaaclab` | [`isaaclab train`](isaaclab.md) (rsl_rl PPO) in `$ISAACLAB_PYTHON` | no, subprocess | 1 RTX GPU, 8 GB |
| `mock` | writes a stub checkpoint and a job record | yes | none |
| `ppo`, `fast_sac`, `fast_td3` | from-scratch actor-critic loops over `SimEnv` / `VecSimEnv` | yes, torch | CPU is declared sufficient for PPO on MuJoCo |

`hardware_floor` is advisory; `validate` checks feasibility against it and never launches.

## TrainSpec

One dataclass for every supervised backend: `dataset_root`, `dataset_repo_id`, `streaming`, `base_model`, `output_dir`, `embodiment`, `steps=10_000`, `global_batch_size=32`, `learning_rate`, `save_freq=1_000`, `num_gpus=1`, `num_nodes=1`, `resume`, `seed`, `method="full"` (`lora`, `expert_only`), `lora_r`, `lora_alpha`, `lora_target_modules`, `tune` (component toggles), `val_episodes`, `augmentation`, `fps`, `extra` (backend-native passthrough). `RLTrainSpec` adds the reward-driven fields ([rl](rl.md)).

## Lifecycle

```python title="sketch"
problems = trainer.validate(spec)            # pure preflight, launches nothing, empty list = launchable
trainer.prepare(spec)                        # optional; cosmos3 converts the base checkpoint here
result = trainer.train(spec)                 # TrainResult: status, job_id, checkpoint_dir, exported_model, metrics, message
trainer.status(result.job_id)                # the "RUNNING != learning" verdict from the run's logs
artifact = trainer.export(spec, result.checkpoint_dir)   # a path create_policy can load
```

`TrainResult.status` is `success`, `running` or `error`; `metrics` (`latest_loss`, `latest_step`, `learning`, `liveness_ok`) tells a live process from a learning one.

## From an agent

`train_policy(action="train" | "validate" | "status" | "export" | "list", provider=..., ...)` mirrors `TrainSpec` and returns the same verdicts. `lerobot_train(action="start" | "status" | "stop" | "list", ...)` runs `lerobot-train` detached, outliving the agent turn ([lerobot](lerobot.md), [tool reference](../../reference/tools.md)).
