---
description: Post-tune a LeRobot policy from a recorded dataset: the lerobot_train tool, the train_policy tool, LerobotTrainer through TrainSpec, and the provider knobs for lerobot, GR00T and Cosmos 3.
---

# LeRobot training

By the end of this page you can turn a recorded LeRobotDataset into a fine-tuned checkpoint that `create_policy` loads back, from Python or from an agent, and you know which knobs are strands' and which belong to lerobot.

```bash
pip install 'strands-robots[lerobot]'    # lerobot[feetech,dataset]; accelerate for train(), peft for method="lora"
```

## Python: TrainSpec and LerobotTrainer

`LerobotTrainer` builds a typed `lerobot.configs.train.TrainPipelineConfig` and calls lerobot's `train(cfg)` in this interpreter. The training logic is lerobot's; the adapter translates the spec, manages resume, and parses the run for a verdict. Needs a GPU for anything past a smoke test.

```python title="sketch"
from strands_robots.training import TrainSpec, create_trainer

spec = TrainSpec(
    dataset_root="~/.cache/huggingface/lerobot/me/so101_pick",   # has meta/info.json; what stop_recording writes
    base_model="lerobot/smolvla_base",
    output_dir="/tmp/so101_smolvla",
    steps=20_000,
    global_batch_size=32,
    save_freq=2_000,
    val_episodes=2,                                              # hold out the last 2 episodes, log eval loss
    method="lora", lora_r=16,
    extra={"policy_type": "smolvla", "policy.freeze_vision_encoder": False},
)
trainer = create_trainer("lerobot_local")
assert trainer.validate(spec) == []
result = trainer.train(spec)
print(result.status, result.checkpoint_dir, result.metrics)
```

Then `create_policy("lerobot_local", pretrained_name_or_path=result.checkpoint_dir, embodiment="so101")` runs it; see [lerobot-local](../policies/lerobot-local.md) for the naming rules the checkpoint now carries.

## What strands adds on top of lerobot

lerobot owns the training knobs (what each policy freezes, sample weighting, relative actions, quantile normalisation, streaming) and documents them at huggingface.co/docs/lerobot. Four things are strands':

- `extra` reaches any field of lerobot's config tree by dotted key: `policy.*`, `dataset.*`, `wandb.*`. Values may be typed or text (decoded by lerobot's draccus, so `"false"` is a boolean); `None` clears an optional field; `extra["policy_type"]` picks the architecture (`act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `pi0_fast`, `groot`, `xvla`, ...).
- `method` is strands' tuning strategy, not lerobot's per-policy defaults: `full`, `lora` (peft, needs `lora_r`), `expert_only` (freezes the VLM; only policies whose lerobot config exposes `train_expert_only`, read live). `lora` and `expert_only` are mutually exclusive.
- `validate()` refuses before launch what lerobot would fail on inside the run: exactly one data source (`dataset_root` or `dataset_repo_id`); `val_episodes` on a streamed, multi-task or count-less dataset (lerobot splits by fraction per task); a `codebase_version` the installed lerobot cannot read (it names the converter); quantile-normalised policy types (`pi05`, `molmoact2`) on a dataset without `q01..q99` stats; `extra["relative_actions"]` on a policy other than `groot`, `pi0`, `pi05`, `pi0_fast`; `num_nodes > 1`; missing `accelerate` or `peft`.
- A fresh start clears an empty leftover `output_dir` and nothing else; `resume=True` continues from `<output_dir>/checkpoints/last`.

`extra["reward_model"]` switches the run to a reward model (`sarm`) instead of a policy; `extra["sample_weighting"]` configures RA-BC weighting. Both take friendly field dicts checked against lerobot's live dataclasses.

## From an agent: two tools

`train_policy` is provider-agnostic and in process. Its arguments are `TrainSpec` field for field plus `action` and `job_id`:

```python title="sketch"
train_policy(action="validate", provider="lerobot_local", dataset_root="...", base_model="lerobot/smolvla_base", output_dir="/tmp/ft")
train_policy(action="train", provider="lerobot_local", dataset_root="...", base_model="lerobot/smolvla_base", output_dir="/tmp/ft", steps=20000, method="lora", lora_r=16)
train_policy(action="status", provider="lerobot_local", job_id="...")     # RUNNING != learning
train_policy(action="export", provider="lerobot_local", output_dir="/tmp/ft")
train_policy(action="list")
```

`lerobot_train` is the detached alternative: it launches `python -m lerobot.scripts.lerobot_train` (or `accelerate launch` for `num_gpus > 1`) as a background process and tracks it in the same on-disk session store as `lerobot_teleoperate`, so the run outlives the agent turn.

| argument | meaning |
|---|---|
| `action` | `start` (default), `status` (pid, uptime, log tail), `stop` (SIGTERM then SIGKILL), `list` |
| `dataset_root`, `policy_type="act"`, `pretrained_path`, `output_dir`, `job_name` | what to train on and where |
| `steps=20000`, `batch_size=8`, `save_freq=5000`, `device="cuda"`, `dtype`, `num_gpus=1` | run size and placement |
| `lora`, `lora_r`, `lora_alpha`, `lora_target_modules`, `train_expert_only`, `gradient_checkpointing` | memory-fit levers; `lora` and `train_expert_only` are mutually exclusive |
| `val_episodes` | holds out the last N episodes via `--dataset.eval_split` plus `--eval_steps`; refused on multi-task datasets |
| `resume`, `push_to_hub`, `session_name`, `extra_flags` | resume only when `checkpoints/last` exists; raw `--key=value` passthrough |

The boolean levers are checked, not read by truthiness: `"false"` is refused rather than turning a lever on.

## GR00T and Cosmos 3 knobs

`create_trainer("groot")` calls Isaac-GR00T's `experiment.run` with a `FinetuneConfig`. `base_model` and `embodiment` are required; `tune` toggles components (`llm`, `visual`, `projector`, `diffusion`; default `{"llm": False, "visual": False, "projector": True, "diffusion": True}`); `augmentation` and `fps` map to GR00T's data config; `extra["groot_root"]` points at the checkout when it is not importable. `num_gpus > 1` uses torch `elastic_launch`.

`create_trainer("cosmos3")` needs `prepare(spec)` first: it converts the HF checkpoint to DCP with `convert_model_to_dcp`. `train` loads a TOML recipe and applies `extra` as Hydra `key.path=value` overrides; `extra["cosmos_root"]` names the checkout. The floor is eight 80 GB GPUs.

`create_trainer("sagemaker", image_uri=..., role_arn=..., instance_type="ml.g5.xlarge")` ships the same `TrainSpec` to a managed job and waits; the image packages one of the trainers above.

## Limits

- `LerobotTrainer` and `train_policy` are single-node; `num_nodes > 1` is refused. Use `sagemaker` or `lerobot_train` with `accelerate` for scale-out.
- The trainer imports lerobot in this process, which pins `transformers>=5`; the in-process `groot` trainer needs `transformers==4.57.3`. Keep them in separate environments.
- No training runs on this documentation machine; every fence above is a sketch by design.
