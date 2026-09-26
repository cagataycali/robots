---
description: Post-tune any policy natively with the Trainer abstraction - one interface over LeRobot, Isaac-GR00T, and Cosmos3 pipelines.
---

# Training

`strands-robots` post-tunes policies through the `Trainer` abstraction - the
training-side peer of [`Policy`](../policies/overview.md), selected by the
**same provider name** you use for inference:

```python
from strands_robots.training import create_trainer, TrainSpec

trainer = create_trainer("lerobot_local")   # same name as create_policy(...)
spec = TrainSpec(
    dataset_root="/tmp/my_dataset",          # what Robot.stop_recording() writes
    base_model="lerobot/act_aloha_sim",
    output_dir="/tmp/ft_out",
    steps=20000,
)
result = trainer.train(spec)                 # -> launches lerobot_train
# result.checkpoint_dir loads straight back into create_policy(...)
```

## Why an abstraction (not just `lerobot train`)

Each provider ships its own post-training pipeline - `lerobot_train` with
draccus flags, Isaac-GR00T's `launch_finetune.py`, cosmos-framework's TOML
recipes, SageMaker's managed job - and one `--policy.type` flag cannot express
them. Every `Trainer` runs the same `validate() -> prepare() -> train() ->
export()` lifecycle plus `status()` for an in-flight job. A local `train()`
blocks and returns a terminal result; a submitted run can return `running` with
a `job_id`, so branch on all three `TrainResult.status` values.

## The data loop, end to end

```bash
pip install "strands-robots[sim-mujoco,lerobot]" "lerobot[training]"   # training needs accelerate
```

```python
import os

from strands_robots import Robot, MockPolicy, create_policy
from strands_robots.training import create_trainer, TrainSpec

if __name__ == "__main__":   # lerobot's DataLoader workers re-import this file (macOS spawn)
    # 1. RECORD - one episode is enough to smoke-test the loop
    sim = Robot("so100", mesh=False)
    sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
    sim.start_recording(repo_id="local/demo", root="/tmp/demo_ds",
                        # the rollout below adopts this fps (it passes no control_frequency)
                        fps=50, task="pick up the red cube", overwrite=True,
                        cameras=["front"])   # else the built-in overview camera is recorded too
    sim.run_policy(robot_name="so100", policy_object=MockPolicy(),
                   instruction="pick up the red cube", n_steps=60)
    sim.stop_recording()        # writes a LeRobotDataset v3 at /tmp/demo_ds

    # 2. TRAIN - thin wrapper over lerobot_train; ACT from scratch on CPU
    trainer = create_trainer("lerobot_local", device="cpu")
    spec = TrainSpec(dataset_root="/tmp/demo_ds", base_model="",
                     output_dir="/tmp/demo_ft", steps=2, save_freq=2,
                     global_batch_size=2, extra={"policy_type": "act"})
    result = trainer.train(spec)

    # 3. EXPORT - loadable artifact (HF-native passthrough for lerobot/groot)
    ckpt = trainer.export(spec, result.checkpoint_dir)

    # 4. DEPLOY - load the freshly-trained checkpoint back as a Policy;
    #    lerobot_local is behind the trust-remote-code gate even for a local dir
    os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")
    policy = create_policy(ckpt, device="cpu")
    sim.run_policy(robot_name="so100", policy_object=policy,
                   instruction="pick up the red cube", n_steps=15)
```

Three details the snippet carries on purpose: the `[training]` extra, because
`lerobot_local` needs `accelerate` on CPU as well as GPU (see
[Dependencies & extras](#dependencies-extras-per-provider)); the
`__main__` guard, because the trainer's DataLoader workers re-import the
script they were started from and would otherwise re-run the recording; and
`cameras=["front"]`, because a recording with no camera list captures every
camera the world has, including the built-in overview one.

Swap `create_trainer("lerobot_local")` → `"groot"` or `"cosmos3"` and **only the
provider string changes** - exactly how `Robot("so100", mode="real")` swaps
sim↔hardware.

## TrainSpec - one spec, many backends

Each trainer reads the fields it supports and ignores the rest; provider knobs
go in `extra` (see [Provider knobs](provider-knobs.md)). `validate()` returns
every problem below as text instead of launching.

| Field | Type | Default | `validate()` refuses |
|-------|------|---------|----------------------|
| `dataset_root` / `dataset_repo_id` | `str` | `""` / `None` | path traversal; a Hub id plus `val_episodes` without a local `meta/info.json` |
| `base_model` | `str` | `""` | leading `-`; empty for GR00T / Cosmos3 |
| `output_dir` | `str` | `""` | path traversal, protected directories |
| `steps` / `global_batch_size` | `int` | `10000` / `32` | zero, negative, fractional, `bool` |
| `learning_rate` | `float \| None` | `None` | non-positive or non-finite |
| `save_freq` | `int` | `1000` | not a positive step count |
| `num_gpus` / `num_nodes` | `int` | `1` | zero, negative, fractional, `bool` |
| `resume` / `streaming` | `bool` | `False` | a non-`bool`; `streaming` together with `val_episodes` |
| `seed` | `int \| None` | `None` | negative |
| `method` | `str` | `"full"` | anything the selected backend cannot forward - LeRobot takes `full` / `lora` / `expert_only`, GR00T `full` / `frozen_backbone` (name components with `tune` instead), Cosmos3 `full` only |
| `lora_r` / `lora_alpha` | `int \| None` | `None` | non-positive |
| `tune` | `dict[str, bool]` | `{}` | keys outside `llm` / `visual` / `projector` / `diffusion`; a non-`bool` value; a policy whose config has no such switches (GR00T via `groot` or `lerobot_local` `policy_type="groot"`) |
| `embodiment` | `str \| None` | `None` | a policy whose config has no embodiment tag (only GR00T declares one; others take their shape from the dataset) |
| `val_episodes` | `int \| None` | `None` | non-positive; multi-task or streamed dataset |
| `extra` | `dict[str, Any]` | `{}` | key not lowercase, or with `-` / `=` / whitespace |

`validate()` refuses a field before anything loads rather than reading it loosely: posture flags (`streaming`, `resume`, each `tune` switch) must be real booleans, counts (`steps`, `global_batch_size`, `val_episodes`, `num_gpus`, `num_nodes`) positive integers, `seed` a non-negative integer, and `val_episodes` needs a single-task dataset whose episode count is readable locally.

## From an agent (natural language)

```python
from strands import Agent
from strands_robots import Robot
from strands_robots import train_policy

agent = Agent(tools=[Robot("so100", mesh=False), train_policy])
agent("Record 50 cube-pick episodes, then post-tune lerobot ACT on the dataset "
      "at /tmp/demo_ds into /tmp/demo_ft, and tell me if it's actually learning.")
```

`train_policy` actions: `train`, `validate`, `status`, `export`, `list`. A
finished run names the checkpoint to load; a still-running managed job names
the `status` poll for its `job_id`; a run with no checkpoint says so.

## Dependencies & extras (per provider)

**Every `lerobot_local` row below also needs `lerobot[training]`, on CPU as well
as GPU**: LeRobot's `train()` calls
`require_package("accelerate", extra="training")` before it branches on
device, and the `[lerobot]` extra is exactly `lerobot[feetech,dataset]`.
`validate()` reports an absent `accelerate` (or `peft` for `method="lora"`) as
a preflight problem and `train()` fails closed, leaving `output_dir` untouched;
the cause is in `TrainResult.message`, not a raise.

```bash
pip install "lerobot[training]"
```

| Provider / policy | Install | Notes |
|---|---|---|
| `lerobot_local` + ACT / diffusion | `pip install 'strands-robots[lerobot]' 'lerobot[training]'` | `[lerobot]` supplies torch, torchcodec, datasets |
| `lerobot_local` + `smolvla` | `pip install 'strands-robots[smolvla]' 'lerobot[training]'` | lerobot's `[smolvla]` extra: `transformers>=5.4.0,<5.6.0` + num2words |
| `lerobot_local` + `pi0` / `pi05` | `pip install 'strands-robots[lerobot]' 'lerobot[training]' 'lerobot[pi]'` | same transformers range + scipy |
| `groot` | Isaac-GR00T checkout, `pip install -e` into the **same** environment | `GR00T_ROOT` / `extra["groot_root"]` = the checkout |
| `cosmos3` | cosmos-framework checkout (`uv sync --group=cu130-train`), same environment | `COSMOS_ROOT` / `extra["cosmos_root"]` = the checkout |

Pin `torch` and `torchcodec` together (verified: `torch==2.10.0+cu128` +
`torchcodec==0.10.0`): a mismatched build load-fails with `undefined symbol:
...MessageLogger` and lerobot swallows the per-shard error. Every local trainer
imports its backend into the interpreter that imports `strands_robots` - there
is no `python_executable=`; a missing package reports
`<package> is not importable from this interpreter`.

## See also

- [Recording](../recording.md) - produce the dataset.
- [Provider knobs](provider-knobs.md) - what each backend accepts through `extra`.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
