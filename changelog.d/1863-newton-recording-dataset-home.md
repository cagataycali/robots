### Fixed: the Newton backend records where the dataset home says, not where `~/.cache` says

`NewtonSimEngine.start_recording` resolved the on-disk dataset directory itself
instead of through the shared `resolve_dataset_dir()` that `DatasetRecorder.create()`
and the MuJoCo and Isaac backends use. Its copy matched the resolver on the two
path-shaped `repo_id` branches and hard-coded `~/.cache/huggingface/lerobot` on
the third, so it ignored `$HF_LEROBOT_HOME` - the override LeRobot honours and
the only way to relocate the dataset home.

Every consumer of the resolved value then read a directory the session does not
write to. With the home relocated and `repo_id="user/ds"`: `overwrite=True`
removed the dataset under the *default* home, which the call never addressed,
while leaving the addressed one for `create()` to remove; the resume probe missed
an existing dataset, so appending an episode dead-ended in a `FileExistsError`
whose two remedies both leave `start_recording` (discard the episodes, or call
`DatasetRecorder.resume()` directly); and `last_dataset_root` - synced by
`stop_recording(bucket=...)` and read by `verify_dataset_episodes` once the
recorder is dropped - named the stale path.

All three backends' `start_recording` are now pinned structurally to resolve
through the shared helper and to spell no component of the default home, so a
backend added later cannot reintroduce a private copy.
