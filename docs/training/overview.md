---
description: Post-tune any policy natively with the Trainer abstraction - one interface over LeRobot, Isaac-GR00T, and Cosmos3 pipelines.
---

# Training

`strands-robots` post-tunes policies **natively** through the `Trainer`
abstraction - the training-side peer of [`Policy`](../policies/overview.md)
(inference). One interface wraps three genuinely different upstream pipelines,
selected by the **same provider name** you use for inference:

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

Not everything is LeRobot. Each backend ships its own post-training pipeline,
and a single `--policy.type` flag can't express them:

| Provider | Upstream entry point | Config surface | Launcher | HW floor |
|----------|---------------------|----------------|----------|----------|
| `lerobot_local` | `lerobot.scripts.lerobot_train` | draccus `--dotted.flags` | `python` / `accelerate launch` | 1 consumer GPU |
| `groot` | Isaac-GR00T `launch_finetune.py` | `FinetuneConfig` (tyro) + `tune_*` flags | `python` / `torchrun` | 1 modern GPU |
| `cosmos3` | `cosmos_framework.scripts.train` | TOML recipe + Hydra overrides; **DCP convert** + **safetensors export** | `torchrun` (HSDP) | 8×H100 80GB |

The `Trainer` ABC hides all of that behind one lifecycle:

```
validate()  ->  prepare()  ->  train()  ->  export()
                   ▲                           ▲
            (cosmos: DCP convert,        (cosmos: DCP -> safetensors;
             groot: modality cfg)         lerobot/groot: passthrough)
```

plus `status()` for a "RUNNING ≠ learning" verdict on an in-flight job.

## The data loop, end to end

```python
from strands_robots import Robot, MockPolicy, create_policy
from strands_robots.training import create_trainer, TrainSpec

# 1. RECORD - one episode is enough to smoke-test the loop
sim = Robot("so100", mesh=False)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
sim.start_recording(repo_id="local/demo", root="/tmp/demo_ds",
                    fps=30, task="pick up the red cube", overwrite=True)
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

# 4. DEPLOY - load the freshly-trained checkpoint back as a Policy
policy = create_policy(ckpt, device="cpu")
sim.run_policy(robot_name="so100", policy_object=policy,
               instruction="pick up the red cube", n_steps=15)
```

Swap `create_trainer("lerobot_local")` → `"groot"` or `"cosmos3"` and **only the
provider string changes** - exactly how `Robot("so100", mode="real")` swaps
sim↔hardware.

## TrainSpec - one spec, many backends

`TrainSpec` carries provider-agnostic fields; each trainer reads what it
supports and **ignores the rest** (the same tolerance rule as
`Policy.get_actions(**kwargs)`). Backend-specific knobs go in `extra`:

| Field | Meaning | Notes |
|-------|---------|-------|
| `dataset_root` | LeRobotDataset v3 root | a data source; has `meta/info.json` (optional when `dataset_repo_id` is set) |
| `dataset_repo_id` | Hub dataset id `org/name` | alternative data source; train from the Hub (lerobot) |
| `streaming` | stream frames, no full materialize | lerobot `StreamingLeRobotDataset`; bounded disk (Hub) / RAM (local) |
| `base_model` | HF id / local ckpt to tune from | required for GR00T & Cosmos |
| `method` | `full` \| `lora` \| `expert_only` \| `frozen_backbone` | `lora`+`expert_only` are mutually exclusive |
| `tune` | `{llm,visual,projector,diffusion}` | GR00T only |
| `val_episodes` | hold out the LAST N episodes | deterministic split |
| `num_gpus` / `num_nodes` | multi-GPU / multi-node | selects the launcher |
| `extra["policy_type"]` | lerobot `--policy.type` | act/diffusion/smolvla/pi0/pi05/... |
| `extra["groot_root"]` | Isaac-GR00T checkout | GR00T |
| `extra["sft_toml"]` / `extra["cosmos_root"]` | recipe + checkout | Cosmos |

## From an agent (natural language)

The `train_policy` tool exposes the abstraction to a Strands Agent:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import train_policy

agent = Agent(tools=[Robot("so100", mesh=False), train_policy])
agent("Record 50 cube-pick episodes, then post-tune lerobot ACT on the dataset "
      "at /tmp/demo_ds into /tmp/demo_ft, and tell me if it's actually learning.")
```

`train_policy` actions: `train`, `validate`, `status`, `export`, `list`.

## Provider-specific knobs

### LeRobot (`lerobot_local`)

```python
TrainSpec(..., method="lora", lora_r=16, extra={"policy_type": "pi05"})
# -> lerobot_train --peft.method_type=LORA --peft.r=16 --policy.type=pi05
```

#### Streaming a large Hub dataset (no full download)

Real datasets (BitRobot / HIW-500, ~50-500 GB) do not fit on a single edge node.
Point the trainer at a Hub dataset id and stream it - lerobot pulls shards on
the fly via `StreamingLeRobotDataset`, so disk stays bounded and the first
forward pass starts without waiting for a full download:

```python
TrainSpec(
    dataset_repo_id="org/hiw_500",   # train from the Hub, not a local root
    streaming=True,                  # -> --dataset.streaming=true
    base_model="lerobot/act_aloha_sim",
    output_dir="/tmp/ft_out",
    extra={"policy_type": "act"},
)
# -> lerobot_train --dataset.repo_id=org/hiw_500 --dataset.streaming=true ...
```

`dataset_root` is optional here - if given it is used as a local cache root.
`streaming=True` also works with a local `dataset_root` (streams from disk with
bounded RAM). Held-out `val_episodes` splitting needs a local `meta/info.json`
to count episodes, so it is a no-op when streaming a Hub dataset with no local
cache (the full Hub dataset is used).

### GR00T (`groot`)

```python
TrainSpec(..., embodiment="GR1",
          tune={"llm": False, "visual": False, "projector": True, "diffusion": True},
          extra={"groot_root": "/path/to/Isaac-GR00T"})
# -> launch_finetune.py --embodiment_tag=GR1 --tune_projector=true ...
```

### Cosmos3 (`cosmos3`)

```python
TrainSpec(..., num_gpus=8,
          extra={"cosmos_root": "/path/to/cosmos-framework",
                 "sft_toml": "examples/toml/sft_config/action_policy_droid_repro.toml"})
# prepare(): convert_model_to_dcp ; train(): torchrun ... --sft-toml=... ;
# export(): DCP -> safetensors
```

## Dependencies & extras (per provider)

The base `strands-robots[lerobot]` extra is enough for **ACT / diffusion from
scratch**, but VLA post-tunes pull in policy-specific stacks. Install the extra
that matches your `extra["policy_type"]` / provider — verified on an L40S GPU:

| Provider / policy | Install | Notes |
|---|---|---|
| `lerobot_local` + ACT / diffusion | `pip install 'strands-robots[lerobot]'` | works out of the box (torch + torchcodec + datasets) |
| `lerobot_local` + `smolvla` | `pip install 'lerobot[smolvla]==0.5.1'` | **needs `transformers==5.3.0`** (lerobot's pinned `[smolvla]` extra). A newer transformers (e.g. 5.12) crashes smolvla import with `non-default argument 'backbone_cfg' follows default argument`. Pin it. |
| `lerobot_local` + `pi0` / `pi05` | `pip install 'lerobot[pi]==0.5.1'` | same `transformers==5.3.0` pin via lerobot's `[pi]` extra |
| `groot` | Isaac-GR00T checkout + its own venv (`omegaconf`, `tyro`, …); point `extra["groot_root"]` / `GR00T_ROOT` at it | launched as a subprocess, so it uses GR00T's interpreter, not ours |
| `cosmos3` | cosmos-framework checkout (`uv sync --group=cu130-train`); point `extra["cosmos_root"]` / `COSMOS_ROOT` at it | torchrun-driven; same subprocess-interpreter rule |

> **torchcodec / torch ABI:** the lerobot training dataloader decodes video via
> `torchcodec`, whose compiled `.so` must match the **exact** installed torch
> build. A torch *nightly* (e.g. `2.12.0.dev`) load-fails a stable-built
> torchcodec with `undefined symbol: ...MessageLogger` even when ffmpeg is
> present — and lerobot silently swallows the per-shard decode error, so
> training fails with a generic non-zero exit. Pin `torch` + `torchcodec`
> together (verified-good combo: `torch==2.10.0+cu128` + `torchcodec==0.10.0`).

> **Subprocess interpreter:** `LerobotTrainer` / `Gr00tTrainer` / `Cosmos3Trainer`
> accept a `python_executable=` argument (defaults to `sys.executable`). Set it
> to a venv that has the provider's deps if your agent process runs in a
> different environment — the training pipeline runs in that interpreter.

## See also

- [Recording](../recording.md) - produce the dataset.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
