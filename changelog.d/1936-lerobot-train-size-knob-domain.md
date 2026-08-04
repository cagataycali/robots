### Fixed: `lerobot_train` refuses a run size it cannot honor, before launching it

`build_train_command` wrote `--steps` and `--batch_size` into the argv verbatim.
Because the tool launches that argv as a detached process, an unusable value was
never reported to the caller - the tool returned a pid and only the training log
recorded, minutes later, that the run could not proceed. Two values were worse
than late: `steps=0` and `steps=-5` parse as valid ints and make lerobot's
training loop (`for _ in range(step, cfg.steps)`) empty, so the run completed
having trained nothing, and `steps=True` / `batch_size=True` parse as `1`, so a
boolean silently became a one-step or batch-of-one run.

Both knobs are now checked against the shared positive-count domain, which also
covers the fractional and non-numeric values lerobot's `int` fields cannot
decode. The refusal reaches the caller as an error envelope before any process
starts. `None` still means "omit the flag and keep lerobot's own default", and
`save_freq` is unchanged - lerobot documents a non-positive value there as
"disables periodic saving", which is a mode selector rather than an unusable
size.

This also removes a divergence: `LerobotTrainer.validate` already refused a
non-positive `steps` for the same lerobot run, so one parameter had two
contracts depending on which surface built the flag.
