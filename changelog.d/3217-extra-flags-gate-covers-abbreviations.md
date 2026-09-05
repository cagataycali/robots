### Fixed

- `lerobot_train`: an `extra_flags` key is now gated as the flag the trainer's
  parser resolves it to, on both axes that parser uses. draccus over `argparse`
  honors any unambiguous prefix, so `{"ou": "/anywhere"}` reached
  `--output_dir` and `{"co": "x"}` reached `--config_path` while the whole-key
  blocklist saw a name on no list. A key is also truncated at
  its first `=` before it is matched, because the emitter appends its own
  (`f"--{key}={value}"`) and `argparse` reads the option name from the text
  before the first `=`: `{"output_dir=/evil/dir": "x"}` emitted
  `--output_dir=/evil/dir=x` and set `output_dir`, so every gated spelling had
  an ungated `=`-carrying twin. Measured against the 713 options a lerobot
  `TrainPipelineConfig` registers, 98 argv spellings reach one of the 8 gated
  flags lerobot declares - 49 prefixes and their 49 value-bearing twins - and 90
  of them passed the gate with no operator prompt, including every spelling of
  `policy.pretrained_path` and `config_path`. None do now. One
  `STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=<flag>`
  entry clears every spelling of the flag it names on both axes - the operator
  approves a flag, not an argv.
