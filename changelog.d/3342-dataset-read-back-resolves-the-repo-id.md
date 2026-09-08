### Fixed: a read reopens the directory the `repo_id` recorded to

Every writing entry point resolves the dataset directory through
`resolve_dataset_dir` and hands LeRobot the result as an explicit `root`,
because a `repo_id` that is itself a path (no `owner/name` slash, or
`./`-prefixed) is a local directory here and is not one to LeRobot, which
resolves any absent root to `$HF_LEROBOT_HOME/{repo_id}` whatever the id looks
like. `load_lerobot_episode` -- the shared loader behind
`Simulation.replay_episode` -- forwarded the caller's `root` unresolved, so a
read by the recording id looked where the recording had never been and the miss
fell through to a Hub lookup for a name that only ever meant a directory.
Measured on lerobot 0.6.2: `start_recording(repo_id="sim_recording", root=None)`
plus a 45-step rollout wrote 7 files under `./sim_recording` with every call
`status="success"`, and `replay_episode(repo_id="sim_recording")` then returned
`status="error"`, "Cannot reach
https://huggingface.co/api/datasets/sim_recording/refs". The read now applies
the same rule and replays 45/45 frames from that directory. Only that rule: an
`owner/name` id keeps its absent root, which is how LeRobot selects the
revision-safe Hub snapshot cache for a download, and is already the directory a
local recording under that id wrote to. The rule those surfaces share is now one
named function, `local_dataset_dir` (#3342).
