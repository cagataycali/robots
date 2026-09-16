### Fixed: through the agent tool, an omitted rate follows the one already in force

`run_policy` / `start_policy` / `run_multi_policy` / `eval_policy` /
`evaluate_benchmark` called with no `control_frequency` while a recording is
open run at the recording's `fps`, and `start_recording` with no `fps` while
one rollout is in flight opens at that rollout's rate — the reply names the
adopted rate. The documented record-then-rollout sequence now succeeds on
the first call instead of failing on the colliding defaults (30 fps vs
50 Hz). A rate the caller passed is still refused on a mismatch.
