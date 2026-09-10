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
| `lerobot_local` | `lerobot.scripts.lerobot_train` | draccus `--dotted.flags` | `python` / `accelerate launch` | CPU for a toy run; 1 consumer GPU in practice |
| `groot` | Isaac-GR00T `launch_finetune.py` | `FinetuneConfig` (tyro) + `tune_*` flags | `python` / `torchrun` | 1 modern GPU |
| `cosmos3` | `cosmos_framework.scripts.train` | TOML recipe + Hydra overrides; **DCP convert** + **safetensors export** | `torchrun` (HSDP) | 8×H100 80GB |
| `sagemaker` | none - the container image's own trainer | the same `TrainSpec`, as job hyperparameters | `CreateTrainingJob` (managed) | none locally; the job brings its own |

Those three local backends import a training library and drive it in-process.
`sagemaker` is the other shape: pure transport, importing no training library and
submitting the spec to a managed runner whose image packages one of the local
paths. The difference is visible in one place a caller has to handle - a local
`train()` blocks and returns a terminal result, while a submitted run can outlive
the call and come back as `running` with a `job_id` to poll via `status()`, and
no checkpoint yet. Branch on all three `TrainResult.status` values, not on
"not `error`".

The `Trainer` ABC hides all of that behind one lifecycle:

```
validate()  ->  prepare()  ->  train()  ->  export()
                   ▲                           ▲
            (cosmos: DCP convert,        (cosmos: DCP -> safetensors;
             groot: modality cfg)         lerobot/groot: passthrough)
```

plus `status()` for a "RUNNING ≠ learning" verdict on an in-flight job.

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

`TrainSpec` carries provider-agnostic fields; each trainer reads what it
supports and **ignores the rest** (the same tolerance rule as
`Policy.get_actions(**kwargs)`). Backend-specific knobs go in `extra`:

| Field | Meaning | Notes |
|-------|---------|-------|
| `dataset_root` | LeRobotDataset v3 root | a data source; has `meta/info.json` (optional when `dataset_repo_id` is set) |
| `dataset_repo_id` | Hub dataset id `org/name` | alternative data source; train from the Hub (lerobot) |
| `streaming` | stream frames, no full materialize | lerobot `StreamingLeRobotDataset`; bounded disk (Hub) / RAM (local); mutually exclusive with `val_episodes`. A posture flag, so `validate()` requires a boolean rather than reading it by truthiness: `"false"` is truthy and would stream, and beside `val_episodes` it was refused with "set streaming=False" at a caller who had spelled exactly that |
| `resume` | continue from the last checkpoint under `output_dir` | lerobot, GR00T, SageMaker. A posture flag, checked like `streaming`: on lerobot a truthy `resume` swaps the spec-built config for the checkpoint's own, so `"false"` would silently drop `steps`, `global_batch_size` and `save_freq` from the run it was meant to start fresh |
| `base_model` | HF id / local ckpt to tune from | required for GR00T & Cosmos |
| `steps` / `global_batch_size` | the run size: optimizer steps x batch | each must be a positive integer; `validate()` refuses `0`, a fractional or non-finite value, and a `bool` (`True` would read as a silent one-step run) before anything is loaded |
| `method` | `full` \| `lora` \| `expert_only` \| `frozen_backbone` | `lora`+`expert_only` are mutually exclusive |
| `tune` | `{llm,visual,projector,diffusion}` | GR00T, either route: the `groot` provider's `--tune_*` flags or `lerobot_local` with `policy_type="groot"`, whose `GrootConfig` declares the same four switches. A key naming no component (`vision` for `visual`), or a component the policy cannot freeze, is refused - an unforwarded toggle trains the config default, which looks exactly like never having asked. Each value must be a `bool`: `"false"` is truthy and is refused rather than read as `True` |
| `embodiment` | which state/action projector head trains | GR00T, either route (`--embodiment_tag` / `GrootConfig.embodiment_tag`). Every other lerobot policy takes its state/action shape from the dataset features and has no such field, so the request is refused rather than dropped |
| `val_episodes` | hold out the LAST N episodes | deterministic split; must be a positive integer below the dataset's episode count, and that count must be readable from a local `meta/info.json` (see the Hub-source note below). `validate()` refuses `0` or a negative (they produced no split and no eval cadence at all - the run trained on everything and logged no validation loss), a `bool`, and a fractional value (`2.7` reserved 3 episodes, `0.5` reserved none while still evaluating); refused on a dataset whose `total_tasks` declares more than one task, or declares something that is not a task count, since lerobot's split is a per-task fraction; mutually exclusive with `streaming` |
| `num_gpus` / `num_nodes` | multi-GPU / multi-node | selects the launcher; each must be a positive integer. `validate()` refuses `0`, a negative, a `bool` and a non-finite value (none of them read as greater than one, so the selector would route them to the single-process path and the run would proceed on a topology nobody asked for) and a fractional or integral float (`2.7`, `2.0` - greater than one, so they reach the launcher as the worker count) |
| `seed` | reproducibility seed | must be a non-negative integer; `validate()` refuses a negative (`torch.manual_seed` would take it modulo `2**64`, so `-1` silently becomes `2**64 - 1`), a fractional or non-finite value, and a `bool`. `None` uses the backend's own default |
| `extra["policy_type"]` | lerobot `--policy.type` | act/diffusion/smolvla/pi0/pi05/... |
| `extra["groot_root"]` | Isaac-GR00T checkout | GR00T |
| `extra["sft_toml"]` / `extra["cosmos_root"]` | recipe + checkout | Cosmos |
| `extra["relative_actions"]` | train pi0-family with delta actions | lerobot `--policy.use_relative_actions=true` (pi0/pi05/pi0_fast) |
| `extra["sample_weighting"]` | RA-BC per-sample loss weighting dict | lerobot `cfg.sample_weighting` (`--sample_weighting.*`) |
| `extra["reward_model"]` | train a reward model (`sarm` / `robometer` / `topreward` / `reward_classifier`) instead of a policy | lerobot `cfg.reward_model` (`--reward_model.*`); requires lerobot >= 0.5.2 |

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

`train` reports the step that fits the run it got. A finished run names the
checkpoint to load; a run that has not finished - the managed-job backend
whose job outlives the submitting process, which returns `running` once its
local poll budget expires - names the `status` poll for its `job_id` instead,
and a run that wrote no discoverable checkpoint says so. The tool never offers
`create_policy(<checkpoint_dir>)` for a result whose `checkpoint_dir` is
`None`, because that renders as `create_policy('None')` and raises
`Unknown policy provider: 'None'`. The run's own status always travels verbatim
in the result's `{"json": ...}` block.

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
device, so "no GPU" does not mean "no extra" - and nothing on this path pulls
`accelerate` in: the `[lerobot]` extra is exactly `lerobot[feetech,dataset]`. The
`strands-robots` extras that declare `accelerate` belong to other providers
(`kimodo`, `cosmos3-diffusers`), so an install only has it by accident - `[all]`
does, because it pulls `[kimodo]`; `[lerobot]` alone does not.

```bash
pip install "lerobot[training]"
```

`validate()` reports an absent `accelerate` - and an absent `peft` when
`method="lora"` - as a preflight problem, so `train()` fails closed on it and
leaves `output_dir` untouched rather than clearing it on the way to a run that
cannot start. Either way the cause arrives in the `TrainResult` rather than as a
raise, so check `result.status` and surface `result.message` - it names the
package and the `lerobot[...]` extra that supplies it. An unchecked call hands
whatever consumes `checkpoint_dir` a `None` instead.

The base `strands-robots[lerobot]` extra covers **recording, streaming, and
loading a trained checkpoint**, and with `lerobot[training]` it covers **ACT /
diffusion from scratch**; VLA post-tunes pull in policy-specific stacks on top.
Install the extra that matches your `extra["policy_type"]` / provider — verified
on an L40S GPU:

| Provider / policy | Install | Notes |
|---|---|---|
| `lerobot_local` + ACT / diffusion | `pip install 'strands-robots[lerobot]' 'lerobot[training]'` | `[lerobot]` supplies torch + torchcodec + datasets; it does **not** supply `accelerate` |
| `lerobot_local` + `smolvla` | `pip install 'strands-robots[smolvla]' 'lerobot[training]'` | `[smolvla]` layers lerobot's own `[smolvla]` extra (`transformers>=5.4.0,<5.6.0` + num2words) on top of `[lerobot]`. Do **not** pin `transformers==5.3.0` - it conflicts with lerobot 0.6's transformers floor. |
| `lerobot_local` + `pi0` / `pi05` | `pip install 'strands-robots[lerobot]' 'lerobot[training]' 'lerobot[pi]'` | lerobot 0.6's `[pi]` extra (same `transformers>=5.4.0,<5.6.0` range + scipy) |
| `groot` | Isaac-GR00T checkout, installed with `pip install -e` into the **same** environment as `strands_robots` (it pulls `omegaconf`, `tyro`, …); point `extra["groot_root"]` / `GR00T_ROOT` at the checkout | `gr00t` is imported in the calling interpreter, so it has to be importable there; `GR00T_ROOT` resolves relative configs, not the interpreter |
| `cosmos3` | cosmos-framework checkout (`uv sync --group=cu130-train`), installed into the **same** environment as `strands_robots`; point `extra["cosmos_root"]` / `COSMOS_ROOT` at the checkout | `cosmos_framework` is imported in the calling interpreter; multi-GPU goes through torch's programmatic `elastic_launch`, not a `torchrun` binary |

> **torchcodec / torch ABI:** the lerobot training dataloader decodes video via
> `torchcodec`, whose compiled `.so` must match the **exact** installed torch
> build. A torch *nightly* (e.g. `2.12.0.dev`) load-fails a stable-built
> torchcodec with `undefined symbol: ...MessageLogger` even when ffmpeg is
> present — and lerobot silently swallows the per-shard decode error, so
> training fails with a generic non-zero exit. Pin `torch` + `torchcodec`
> together (verified-good combo: `torch==2.10.0+cu128` + `torchcodec==0.10.0`).

> **One interpreter:** `LerobotTrainer` / `Gr00tTrainer` / `Cosmos3Trainer` call their
> backend as a library in the **same** interpreter that imports `strands_robots`, so
> there is no second interpreter to point them at. There is no `python_executable=`
> argument either, and because each constructor absorbs unknown keywords, passing one
> is silently a no-op rather than an error. Install the provider's deps into the
> environment your agent process runs in; otherwise `train()` reports
> `<package> is not importable from this interpreter` in `TrainResult.message`.
> `GR00T_ROOT` / `COSMOS_ROOT` (and `extra["groot_root"]` / `extra["cosmos_root"]`)
> resolve the checkout so relative configs load - they are not interpreter paths.

## See also

- [Recording](../recording.md) - produce the dataset.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
