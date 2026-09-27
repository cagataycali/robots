---
description: What each training backend accepts - lerobot's config tree through extra, reward models and sample weighting, and the GR00T and Cosmos3 knobs.
---

# Training provider knobs

The fields `TrainSpec.extra` reaches on each backend, and what `validate()`
refuses before a launch. The abstraction itself, the record-train-deploy loop,
the `TrainSpec` table and the per-provider installs are
[Training](overview.md).

## LeRobot (`lerobot_local`)

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
or count-less dataset (lerobot splits by fraction per task); `extra["relative_actions"]`
on any policy other than `groot` / `pi0` / `pi05` / `pi0_fast`.

Those rosters name what the lowest lerobot this package supports accepts. A
newer lerobot inside the supported range can carry a capability on more policy
types than the page lists, and the gate reads your installed lerobot rather than
this page, so it accepts them; the page gains them when the supported floor
moves onto that release.

**Reward models train through the same trainer.** `extra["reward_model"]`
selects a lerobot reward model (`sarm`, `robometer`, `topreward`,
`reward_classifier`) with that type's own fields, and
`extra["sample_weighting"]` (`type`, `progress_path`, `head_mode`, `kappa`,
`epsilon`, plus `extra_params` for a scheme's own knobs) weights a policy run by
RA-BC progress. Neither field list is written down in this trainer: both are read
off the installed lerobot - the reward type's own config fields, and
`SampleWeightingConfig`'s fields - so a field lerobot adds is configurable the day
it lands and the refusal below names the surface as it actually is. Both dicts are
refused before launch for a field the chosen type has no home for, for a `type`
lerobot does not ship, and for the pipeline-ordering mistake of weighting a
reward-model run. A `type` lerobot does not ship is reported on its own: an
unresolved type has no config class, so nothing is claimed about which fields it
takes - correct the name and the field check runs against the real one.
The progress parquet between the two runs is lerobot's to produce:

```bash
python -m lerobot.rewards.sarm.compute_rabc_weights \
    --dataset-repo-id org/cube_pick \
    --reward-model-path /tmp/sarm_out/checkpoints/last/pretrained_model \
    --output-path /tmp/sarm_progress.parquet
```

So the loop is three runs: train the reward model here, compute the parquet
there, then point `extra["sample_weighting"]["progress_path"]` at it. To score
frames with a trained reward model, load it with lerobot's `make_reward_model`.

## GR00T (`groot`) and Cosmos3 (`cosmos3`)

`embodiment` + `tune` + `extra["groot_root"]` drive `launch_finetune.py`; with
`extra={"policy_type": "groot"}` the same two fields reach lerobot's own
`GrootConfig(embodiment_tag=..., tune_projector=...)` instead, discovered off the
config class - see [Isaac-GR00T](../policies/groot.md). GR00T freezes components
individually rather than by strategy, so `method="expert_only"` is refused for it
and the set is named directly: `tune={"projector": False}` trains the diffusion
action head with everything before it frozen. `num_gpus` + `extra["cosmos_root"]` +
`extra["sft_toml"]` drive `prepare()` (DCP convert), `train()` (`torchrun`) and
`export()` (DCP -> safetensors) - see [Cosmos3](../policies/cosmos3.md).

Neither backend takes a LoRA request. GR00T has no config field to carry one and
Cosmos3 writes no adapter override, so `method="lora"` is refused by `validate()`
instead of being run as the full fine-tune the caller did not ask for; on Cosmos3
a different tuning strategy belongs in the recipe TOML (`extra["sft_toml"]`).

## See also

- [Training](overview.md) - the `Trainer` lifecycle, `TrainSpec` and the installs.
- [LeRobot policies](../policies/lerobot-local.md) - the inference peer of these knobs.
- [Isaac-GR00T](../policies/groot.md), [Cosmos3](../policies/cosmos3.md) - the two
  non-lerobot backends.
