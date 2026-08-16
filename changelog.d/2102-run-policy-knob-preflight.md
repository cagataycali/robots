### Fixed: `run_policy` tool checks `control_frequency` / `action_horizon` before it destroys the dataset

The `run_policy` agent tool owns the episode loop and the recording lifecycle,
and it starts that recording with `overwrite=True`. `control_frequency` and
`action_horizon` were the only rollout knobs it forwarded without a pre-flight
check, so a value the facade refuses was discovered inside the episode loop -
after an existing dataset at `dataset_root` had been removed and replaced with
an empty one. Measured over a real MuJoCo rollout that had recorded one episode
of four frames, all 16 unusable values across the two knobs took
`meta/info.json` from `total_episodes=1, total_frames=4` to `0, 0` and reported
`0/1 episodes ok`; the correct reason was already there, buried in every
per-episode record.

Both knobs are now checked alongside the tool's existing `seed` / `video` /
`policy_config` / `stop_when` pre-flight, on the same shared domains the facade
applies (`_validate_positive_frequency` and `_validate_action_horizon` delegate
to `positive_finite_number_error` and `positive_count_error`). The reported
message is byte-identical for every probe value - only its timing changes, from
after the destruction to before it - and their `Args:` entries now state the
domain, which is what the agent tool schema shows a model.
