### Fixed: the `run_policy` tool stops on an episode the recorder could not flush

`PolicyRunner._finalize_recorder_episode` reports a failed `save_episode` by
returning the reason - the recorder marks itself closed, so every later episode's
frames reach no dataset and are not counted as drops either. The `run_policy`
tool, which owns the `n_episodes` loop precisely so no count is self-reported,
called that helper and discarded its return: a rollout that lost episode 1 of 3
ran the other two into the closed recorder and reported `3/3 episodes ok` with
three MP4s, while the reason it held was never logged or returned. Its one
warning then named the wrong cause, reporting the boundary as one that "did not
fire as expected" when it had fired and said why it failed. The loop now stops at
that episode and reports the reason as `recording_save_error`, the posture
`PolicyRunner.evaluate`, `save_episode`, `stop_recording` and `reset` already
take; `n_episodes_ok` and `video_paths` cover only the episodes that ran, and the
count-mismatch guard names the flush. Being unable to *reach* the boundary (no
`PolicyRunner` import, a construction failure) stays tolerated - nothing was
flushed, so no recorder is poisoned.
