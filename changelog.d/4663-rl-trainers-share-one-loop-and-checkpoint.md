### Changed: the three RL trainers run one training loop and write one checkpoint layout

`PpoTrainer`, `FastSacTrainer` and `FastTd3Trainer` each carried their own
copy of `save_checkpoint`, `latest_checkpoint`, `export`, `hardware_floor` and
the observation-normalizer helpers, and SAC and TD3 each carried a copy of the
training loop that differed from `BaseRLAlgo.train` in two lines. They now
inherit all of it from `BaseRLAlgo`: the off-policy warmup is the
`_ready_to_update` hook and SAC's `log_alpha` the `_extra_checkpoint_state`
hook. Checkpoints, metrics and results are unchanged.
