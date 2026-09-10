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

Each backend ships its own post-training pipeline; one `--policy.type` flag
cannot express them.

| Provider | Upstream entry point | HW floor |
|----------|---------------------|----------|
| `lerobot_local` | `lerobot.scripts.lerobot_train` (draccus flags) | CPU for a toy run; one consumer GPU |
| `groot` | Isaac-GR00T `launch_finetune.py` | one modern GPU |
| `cosmos3` | `cosmos_framework.scripts.train` (TOML + DCP convert/export) | 8x H100 |
| `sagemaker` | none - submits the same `TrainSpec` to a managed job | none locally |

Every trainer runs `validate() -> prepare() -> train() -> export()`, plus
`status()` for an in-flight job. A local `train()` blocks and returns a
terminal result; a submitted run can come back `running` with a `job_id` to
poll. Branch on all three `TrainResult.status` values, not on "not `error`".

## The data loop, end to end

```bash
pip install "strands-robots[sim-mujoco,lerobot]" "lerobot[training]"   # training needs accelerate
```

```python
from strands_robots import Robot, MockPolicy, create_policy
from strands_robots.training import create_trainer, TrainSpec

if __name__ == "__main__":   # lerobot's DataLoader workers re-import this file (macOS spawn)
    # 1. RECORD - one episode is enough to smoke-test the loop
    sim = Robot("so100", mesh=False)
    sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
    sim.start_recording(repo_id="local/demo", root="/tmp/demo_ds",
                        # fps must equal the rollout's control_frequency (default 50.0)
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

    # 4. DEPLOY - load the freshly-trained checkpoint back as a Policy
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

Each trainer reads the fields it supports and **ignores the rest**;
backend-specific knobs go in `extra`. `validate()` refuses a non-positive
`steps` / `global_batch_size` / `num_gpus` / `num_nodes` / `val_episodes`, a
negative `seed`, and `streaming` together with `val_episodes`.

| Field | Meaning |
|-------|---------|
| `dataset_root` / `dataset_repo_id` | LeRobotDataset root with `meta/info.json`, or a Hub id `org/name` |
| `streaming` | stream shards instead of materializing the dataset (lerobot) |
| `resume` | continue from the last checkpoint under `output_dir` |
| `base_model` | HF id / local checkpoint to tune from (required for GR00T, Cosmos) |
| `steps` / `global_batch_size` | optimizer steps x batch |
| `method` | `full` \| `lora` \| `expert_only` \| `frozen_backbone` (`lora`+`expert_only` exclusive) |
| `tune` | `{llm,visual,projector,diffusion}` - GR00T, via `groot` or `lerobot_local` with `policy_type="groot"` |
| `embodiment` | which state/action projector head trains - GR00T only; refused for policies that take their shape from the dataset |
| `val_episodes` | hold out the LAST N episodes |
| `num_gpus` / `num_nodes` | selects the launcher |
| `seed` | reproducibility |
| `extra["policy_type"]` | lerobot `--policy.type`: act / diffusion / smolvla / pi0 / pi05 / ... |
| `extra["groot_root"]`, `extra["cosmos_root"]` / `extra["sft_toml"]` | checkout paths and recipe |

`validate()` refuses a field before anything loads rather than reading it loosely: posture flags (`streaming`, `resume`, each `tune` switch) must be real booleans, counts (`steps`, `global_batch_size`, `val_episodes`, `num_gpus`, `num_nodes`) positive integers, `seed` a non-negative integer, and `val_episodes` needs a single-task dataset whose episode count is readable locally.

## From an agent (natural language)

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import train_policy

agent = Agent(tools=[Robot("so100", mesh=False), train_policy])
agent("Record 50 cube-pick episodes, then post-tune lerobot ACT on the dataset "
      "at /tmp/demo_ds into /tmp/demo_ft, and tell me if it's actually learning.")
```

`train_policy` actions: `train`, `validate`, `status`, `export`, `list`. A
finished run names the checkpoint to load; a still-running managed job names
the `status` poll for its `job_id`; a run with no checkpoint says so.

## Provider-specific knobs

### LeRobot (`lerobot_local`)

```python
TrainSpec(..., method="lora", lora_r=16, extra={"policy_type": "pi05"})
# -> lerobot_train --peft.method_type=LORA --peft.r=16 --policy.type=pi05
```

lerobot owns the training knobs - what each policy freezes, RA-BC sample
weighting, relative actions, quantile normalization, dataset streaming, format
versions - and documents them at
[huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot). strands adds
four things on top.

**`extra` reaches any field of lerobot's config tree.** Dotted keys address
sub-configs (`policy.*`, `dataset.*`, `wandb.*`); values may be Python-typed or
text (decoded by lerobot's own draccus decoder, so `"false"` is a boolean, not a
truthy string); `None` clears an optional field; a key that names no field is
ignored with a warning. `method="full"` selects strands' tuning strategy, not
lerobot's per-policy freeze defaults, so full-tuning SmolVLA means saying so:

```python
TrainSpec(
    dataset_repo_id="org/tictactoe",
    base_model="lerobot/smolvla_base",
    output_dir="/tmp/ft_out",
    steps=20000,
    method="full",
    extra={"policy_type": "smolvla",
           "policy.freeze_vision_encoder": False,
           "policy.train_expert_only": False},
)
```

**A fresh start clears an empty leftover `output_dir`** and nothing else; a
directory with contents is left for lerobot to refuse by name, and
`resume=True` continues in place.

**`validate()` refuses before launch** what lerobot would fail on inside the
run: a `policy_type` whose stats want quantiles (`molmoact2`, `pi05`) on a
dataset without `q01..q99`; a `codebase_version` older than the installed
lerobot reads (names the converter); `val_episodes` on a streamed, multi-task,
or count-less dataset (lerobot splits by fraction per task); `use_relative_actions`
on any policy other than `pi0` / `pi05` / `pi0_fast`.

**Reward models train through the same trainer.** `extra["reward_model"]`
selects a lerobot reward model (`sarm`, `robometer`, `topreward`,
`reward_classifier`) with that type's own fields; `compute_rabc_weights`,
`load_reward_model` and `reward_progress` in `strands_robots.training` turn a
trained SARM into the `sample_weighting.progress_path` parquet RA-BC reads.

### GR00T (`groot`)

```python
TrainSpec(..., embodiment="GR1",
          tune={"llm": False, "visual": False, "projector": True, "diffusion": True},
          extra={"groot_root": "/path/to/Isaac-GR00T"})
# -> launch_finetune.py --embodiment_tag=GR1 --tune_projector=true ...
```

lerobot ships its own GR00T port, so the same `embodiment` and `tune` fields
also drive `lerobot_local` - one install, and the resume / LoRA / validation
path every other lerobot policy takes:

```python
TrainSpec(..., embodiment="GR1",
          tune={"llm": False, "visual": False, "projector": True, "diffusion": True},
          extra={"policy_type": "groot"})
# -> cfg.policy = GrootConfig(embodiment_tag="GR1", tune_projector=True, ...)
```

`groot` is the only lerobot policy type whose config declares those fields, and
the trainer discovers that off the config class rather than from a list, so a
policy lerobot adds with an embodiment tag or component toggles is accepted on
arrival and every other policy keeps refusing the request instead of silently
training its defaults.

### Cosmos3 (`cosmos3`)

```python
TrainSpec(..., num_gpus=8,
          extra={"cosmos_root": "/path/to/cosmos-framework",
                 "sft_toml": "examples/toml/sft_config/action_policy_droid_repro.toml"})
# prepare(): convert_model_to_dcp ; train(): torchrun ... --sft-toml=... ;
# export(): DCP -> safetensors
```

## Dependencies & extras (per provider)

**Every `lerobot_local` row below also needs `lerobot[training]`, on CPU as well
as GPU.** LeRobot's `train()` calls
`require_package("accelerate", extra="training")` *before* it branches on
device, and the `[lerobot]` extra is exactly `lerobot[feetech,dataset]`, so
nothing on this path pulls `accelerate` in.

```bash
pip install "lerobot[training]"
```

`validate()` reports an absent `accelerate` (and an absent `peft` for
`method="lora"`) as a preflight problem; `train()` then fails closed and leaves
`output_dir` untouched. The cause arrives in `TrainResult.message`, not as a
raise - check `result.status`.

| Provider / policy | Install | Notes |
|---|---|---|
| `lerobot_local` + ACT / diffusion | `pip install 'strands-robots[lerobot]' 'lerobot[training]'` | `[lerobot]` supplies torch, torchcodec, datasets |
| `lerobot_local` + `smolvla` | `pip install 'strands-robots[smolvla]' 'lerobot[training]'` | layers lerobot's `[smolvla]` extra (`transformers>=5.4.0,<5.6.0` + num2words) |
| `lerobot_local` + `pi0` / `pi05` | `pip install 'strands-robots[lerobot]' 'lerobot[training]' 'lerobot[pi]'` | same transformers range + scipy |
| `groot` | Isaac-GR00T checkout, `pip install -e` into the **same** environment | `extra["groot_root"]` / `GR00T_ROOT` = the checkout |
| `cosmos3` | cosmos-framework checkout (`uv sync --group=cu130-train`), same environment | `extra["cosmos_root"]` / `COSMOS_ROOT` = the checkout |

> **torchcodec / torch ABI:** the training dataloader decodes video via
> `torchcodec`, whose compiled `.so` must match the exact installed torch build;
> a torch nightly load-fails a stable torchcodec with `undefined symbol:
> ...MessageLogger` and lerobot swallows the per-shard error. Pin them together
> (verified: `torch==2.10.0+cu128` + `torchcodec==0.10.0`).

> **One interpreter:** every local trainer imports its backend into the
> interpreter that imports `strands_robots`; there is no `python_executable=`.
> Install the provider's deps into the environment your agent runs in, or
> `train()` reports `<package> is not importable from this interpreter`.
> `GR00T_ROOT` / `COSMOS_ROOT` resolve the checkout, not an interpreter.

## See also

- [Recording](../recording.md) - produce the dataset.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
