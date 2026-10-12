### Tests: RL run size, checkpoint cadence and posture flags join the RL refusal table

`total_timesteps`, `rollout_steps`, `log_interval`, `normalize_obs`,
`normalize_advantage` and `autotune_alpha` are now rows of the RL field table in
`tests/training/test_rl_spec_field_domains.py`, and that table now checks every
training backend, so the supervised ones are also checked to stay silent about
RL-only fields. The suites for those fields keep what each field does to the run
that consumes it. 182 cells become 137, and every planted guard removal still
fails.
