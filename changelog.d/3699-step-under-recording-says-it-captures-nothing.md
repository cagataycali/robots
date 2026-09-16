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

Every rollout that records is named, because `policy_running` - the flag this
guard reads - is raised for each of them: `_announce_rollout` for `run_policy`
and `start_policy`, whose rollouts feed the recorder through the same per-step
hook, and `run_multi_policy`, whose synchronized loop calls `add_frame` itself
and which `describe()` advertises as the path for bimanual data collection.
Naming `run_policy` alone sent a caller who used either of the other two
looking for a defect that was not there.
