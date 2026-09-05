### Fixed

- `lerobot_train`: an `extra_flags` key is now gated as the flag the trainer's
  parser resolves it to, on both axes that parser uses. draccus over `argparse`
  honors any unambiguous prefix, so `{"ou": "/anywhere"}` reached
  `--output_dir` and `{"co": "x"}` reached `--config_path` while the whole-key
  blocklist saw a name on no list - 43 argv spellings reached one of the 8 gated
  flags lerobot registers with no operator prompt. A key is also truncated at
  its first `=` before it is matched, because the emitter appends its own
  (`f"--{key}={value}"`) and `argparse` reads the option name from the text
  before the first `=`: `{"output_dir=/evil/dir": "x"}` emitted
  `--output_dir=/evil/dir=x` and set `output_dir`, so every gated spelling had
  an ungated `=`-carrying twin. One `STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=<flag>`
  entry clears every spelling of the flag it names on both axes - the operator
  approves a flag, not an argv.
