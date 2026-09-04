### Fixed

- `reset()` now closes the open dataset episode on **every** simulation backend, not
  only MuJoCo. `DatasetRecordingMixin.save_episode` documents the boundary as a fact
  about the class all three engines inherit, and `docs/recording.md` repeats it with a
  worked `run_policy` + `reset` loop promising 20 episodes; on Newton and Isaac that
  loop produced **one** episode whose frames spanned every reset teleport in between.
  Nothing reported it - the recorder and the parquet both counted one episode, so
  `stop_recording`'s author-versus-parquet gate had nothing to compare, and
  `verify_dataset_episodes` counts episodes rather than comparing them to the rollouts
  that were run. The rule now has a single owner that all three resets ask, so a
  fourth backend inherits it. A *partial* Isaac reset (`reset(env_ids=[...])`)
  deliberately cuts no boundary, because whether the recorded robot's rollout ended is
  not knowable from `env_ids`.
