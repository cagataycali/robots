### Fixed: `start_recording` says when it resumes an existing dataset

Re-recording a `repo_id`/`root` that already held a dataset appended to it
silently (the resume was a log line; `overwrite` is not a tool parameter),
so the "first" new episode landed at `episode_index 2` unannounced. The
answer now carries a RESUMING line with the episodes and frames already on
disk, the index the next episode gets and how to start fresh, plus a json
block (`resumed`, `episodes_on_disk`, `frames_on_disk`, `dataset_dir`).
