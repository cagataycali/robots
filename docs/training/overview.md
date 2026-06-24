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
| `lerobot_local` | `lerobot.scripts.lerobot_train.train(cfg)` | typed `TrainPipelineConfig` (in-process) | in-process / `accelerate.notebook_launcher` | 1 consumer GPU |
| `groot` | Isaac-GR00T `launch_finetune.py` | `FinetuneConfig` (tyro) + `tune_*` flags | in-process (runpy) / `elastic_launch` | 1 modern GPU |
| `cosmos3` | `cosmos_framework.scripts.train` | TOML recipe + Hydra overrides; **DCP convert** + **safetensors export** | in-process (runpy) / `elastic_launch` (HSDP) | 8×H100 80GB |

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
| `dataset_root` | LeRobotDataset v3 root | required; has `meta/info.json` |
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
# -> build_config() yields a typed TrainPipelineConfig:
#      policy = make_policy_config("pi05"); policy.use_peft = True
#      peft   = PeftConfig(method_type="LORA", r=16)
#    then lerobot.scripts.lerobot_train.train(cfg) is called IN-PROCESS.
```

**Runs in-process - no `subprocess`.** The LeRobot backend imports
`lerobot` directly and calls its `train(cfg)` function in *this*
interpreter. (GR00T and Cosmos3 are also shell-free now - see below - but
they run their upstream *scripts* via `runpy` from a separately-installed
checkout, whereas lerobot is a first-class dependency we call as a library.) `build_config()` translates a `TrainSpec`
into the typed `TrainPipelineConfig` dataclass tree that `train()` consumes;
lerobot's `@parser.wrap()` short-circuits when handed a config instance, so
**`sys.argv` is never read and no command line is assembled**. This removes
the previous attack surface where caller-controlled `extra` keys were
interpolated into a shell `argv` (`--{key}={value}`) for a spawned
interpreter. Unknown `extra` keys are now applied via `setattr` onto the
typed config only when a matching field exists, and ignored (with a warning)
otherwise - they can never become an arbitrary process flag.

Launcher selection stays shell-free:

- **1 GPU / CPU** -> `train(cfg)` called directly (zero new processes).
- **>1 GPU, 1 node** -> `accelerate.notebook_launcher(train, (cfg,),
  num_processes=num_gpus)` (multiprocessing workers, not a command line).
- **multi-node** (`num_nodes > 1`) -> rejected in `validate()` with a clear
  message; genuine multi-node needs a per-node `torchrun`/`accelerate launch`
  that this in-process trainer deliberately does not shell out to.

### GR00T (`groot`)

```python
TrainSpec(..., embodiment="GR1",
          tune={"llm": False, "visual": False, "projector": True, "diffusion": True},
          extra={"groot_root": "/path/to/Isaac-GR00T"})
# -> build_args() yields the flag LIST [--embodiment_tag=GR1,
#    --tune_projector=true, ...]; launch_finetune.py is then run IN-PROCESS
#    via runpy (single GPU) or torch elastic_launch workers (multi-GPU).
#    No subprocess, no torchrun binary. Unsafe extra keys are dropped.
```

### Cosmos3 (`cosmos3`)

```python
TrainSpec(..., num_gpus=8,
          extra={"cosmos_root": "/path/to/cosmos-framework",
                 "sft_toml": "examples/toml/sft_config/action_policy_droid_repro.toml"})
# All three stages run IN-PROCESS via runpy (no subprocess/torchrun binary):
#   prepare(): cosmos_framework.scripts.convert_model_to_dcp
#   train():   cosmos_framework.scripts.train via torch elastic_launch (HSDP)
#   export():  cosmos_framework.scripts.export_model  (DCP -> safetensors)
# Hydra tail overrides from extra are gated by a safe-key allowlist.
```

## See also

- [Recording](../recording.md) - produce the dataset.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
