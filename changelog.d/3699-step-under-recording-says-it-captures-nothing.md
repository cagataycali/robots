### Changed: `step` under an active dataset recording says it captures nothing

The dataset recorder is fed by a policy rollout's per-step hook and by nothing
else. Scripting a demonstration with `set_joint_positions` + `step` under an
active recording captured zero frames, and the caller learned that only from
`stop_recording`'s empty-dataset refusal - after the whole motion had run.

`step` now leads its success text with `NOT RECORDED: a dataset recording is
active but step captures no frames - only a policy rollout feeds the recorder
(start_recording -> run_policy or start_policy -> stop_recording)` when a recording is active
and no rollout is in flight (leading, because a trailing note was read past); a running rollout (which does record) keeps the summary as it
was. `start_recording`'s success text states the same rule where the recording
begins instead of the softer "Run policies to capture frames".

Both launchers are named because both record: `_announce_rollout` raises the
`policy_running` flag this guard reads for `run_policy` and for `start_policy`
alike, and a `start_policy` rollout feeds the recorder through the same hook.
