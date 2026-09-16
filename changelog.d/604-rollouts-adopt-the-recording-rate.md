### Fixed: `start_recording()` then `run_policy()` works on the defaults

The recorder's default is 30 fps and a rollout's was 50 Hz, so the plain
two-call data-collection sequence refused itself every time ("the active
recording declares 30 fps but this rollout captures at control_frequency=50
Hz"). An unset `control_frequency` on `run_policy` / `start_policy` /
`run_multi_policy` / `eval_policy` / `evaluate_benchmark` now adopts the open
recording's fps (else `DEFAULT_CONTROL_FREQUENCY = 50`); a rate you pass is
never corrected, a disagreeing one is still refused. `control_frequency` gains
a description in the tool spec.
