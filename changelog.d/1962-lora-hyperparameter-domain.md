### Fixed: the LoRA adapter hyperparameters are one shared positive-count domain

`lora_r` and `lora_alpha` are the rank and the scaling numerator of a LoRA
fine-tune, and three places wrote them without judging them: the in-process
`peft_kwargs` the LeRobot trainer builds, that trainer's argv-parity command,
and `build_train_command`, a second independent writer of the same two flags.

Only one of the two failed loudly. peft refuses a non-positive `lora_r`, but
only from inside `get_peft_model` once the base model is downloaded and loaded,
and a `bool`/float rank raised out of torch's tensor allocation instead.
`lora_alpha` is a bare numerator nothing compares, so every unusable value was
accepted: `lora_alpha=0` built the adapter, reported its trainable parameters
and trained them with a scaling of `0.0`, so the adapter provably could not
change the model's output -- the run completed, wrote checkpoints, and had
learned nothing that could ever be applied. A negative alpha applied the
negation of what the adapter learned, and `True` was a silent alpha of one.

The two paths also disagreed about a fractional value: peft accepts
`lora_alpha=2.7` in-process and scales by `2.7 / r`, while lerobot's
`PeftConfig` declares both fields `int`, so the argv spelling of the same run
was refused by draccus.

Both fields now go through the same shared `positive_count_error` domain the
run-size knobs use, in one new `Trainer._lora_hyperparameter_problems` gate that
the trainer and the tools-layer writer both apply -- scoped to
`method == "lora"`, since neither field is read under another strategy.
