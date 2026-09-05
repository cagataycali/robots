### Fixed

- `lerobot_train`: an `extra_flags` key that abbreviates a gated flag is now
  gated as that flag. The trainer's parser (draccus over `argparse`) honors any
  unambiguous prefix, so `{"ou": "/anywhere"}` reached `--output_dir` and
  `{"co": "x"}` reached `--config_path` while the whole-key blocklist saw a name
  on no list. 43 argv spellings reached one of the 8 gated flags lerobot
  registers with no operator prompt; the same
  `STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=<flag>` entry now clears every spelling of
  the flag it names.
