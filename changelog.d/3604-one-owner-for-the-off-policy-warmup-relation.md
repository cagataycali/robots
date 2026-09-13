### Fixed

- Both off-policy RL backends carried a verbatim copy of the
  `learning_starts >= batch_size` warmup relation inline in `validate()`, on the
  line before the shared gate that owns the other half of the same contract. The
  relation and the `learning_starts` count domain now live in one place,
  `Trainer._rl_warmup_batch_problems`, so the two backends cannot drift apart on
  a rule they state identically. Reported messages are unchanged.
