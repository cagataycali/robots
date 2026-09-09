### Fixed: a training `resume` or `streaming` that is not a boolean is refused, instead of selecting the other posture

`TrainSpec.resume` and `TrainSpec.streaming` each select a posture rather than
scaling a quantity, and every backend that read them did so by truthiness.
Every non-empty string is truthy, so the spellings a caller reaches for when
opting out selected the affirmative posture. On lerobot, `resume="false"` with
a checkpoint under `output_dir` emitted `--resume=true --config_path=<ckpt>`,
and the in-process `build_config` returns the checkpoint's own config in place
of the spec's - so the fresh run that was asked for became a resume that drops
the caller's `steps`, `global_batch_size` and `save_freq`. `streaming="false"`
beside `val_episodes` was refused with "set streaming=False to keep the
validation split", the remedy the caller had already spelled: the flag gates
that pair check, so the misread posture was reported as the option it selected
rather than as itself. GR00T emitted `--resume_from_checkpoint` and SageMaker
forwarded the raw string as a hyperparameter. The falsy non-booleans (`0`,
`None`, `""`) took the negative posture without being a declared spelling of it,
and nothing reported either direction, because nothing raised.

Both fields are now held to the shared `boolean_flag_error` domain through two
field-scoped gates on `Trainer`, `_resume_problems` and `_streaming_problems`,
in the same biconditional as the numeric domains beside them: a backend that
reads the field routes it through the gate (lerobot, GR00T and SageMaker for
`resume`; lerobot and SageMaker for `streaming`) and a backend that ignores the
field reports nothing about it. Two gates rather than one because the readers
differ. lerobot consults the streaming gate ahead of the pair check and reads
the flag only when it reports nothing, so a misread posture is refused by its
own name. The provider-agnostic `train_policy` tool reaches both through
`validate` on every action that builds a spec. The two honoured postures are
unchanged.
