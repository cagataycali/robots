### Changed: `resolve_dataset_dir`, `local_dataset_dir` and `load_lerobot_episode` moved to `strands_robots.dataset_source`

Import them from `strands_robots.dataset_source` rather than
`strands_robots.dataset_recorder`; there is no re-export, so the three names have
one home. Behaviour is unchanged.

Which directory a `repo_id` names, and where an episode's frames start, are facts
about addressing a dataset, not about writing one -- and the recorder session, the
three simulation recording backends, the rollout runner's `replay` and the
teleoperation tool all have to resolve one id identically, since a second
derivation is a read that misses the write. They sat in `app` with the writer, so
`simulation.policy_runner` deferred an import of the recorder to find out where
its own recording went. `dataset_source` sits in `core` beside
`dataset_metadata`, under all four callers, and its own imports are what let it:
`quiet_video_backend`, `non_negative_whole_number_error`, and LeRobot itself
inside the two functions that open a dataset.

One deferred upward edge leaves the graph (`scripts/check_import_layers.py`: 10
to 9 inversions, `sim|policies -> app` 6 to 5) and its line leaves the declared
roster, so the ratchet fails until the edge is really gone.
