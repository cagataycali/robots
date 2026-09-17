### Fixed: a config that declares its own re-query interval below the trained chunk is named

`_auto_detect_actions_per_step` already warns when a *caller* pins
`actions_per_step` below `config.n_action_steps`, because truncating the chunk
puts every re-query out of distribution. A checkpoint whose config declares
`n_action_steps` below its own `chunk_size` lands in that same regime with no
caller involved, and said nothing - at the extreme `n_action_steps=1` it emitted
no log line at all. Driven that way an ACT checkpoint stalls rather than fails:
re-queried from its own lagging state it converges on commanding where it
already is, so it stops part-way through the motion and reports success. Both
supported exits are now named: `n_action_steps = chunk_size` to replay the chunk
as trained, or `temporal_ensemble_coeff` to consume the whole chunk per step.
Every LeRobot policy ships the two equal, so this cannot fire on an unedited
checkpoint.
