### Fixed: streaming a dataset back reopens the directory the `repo_id` recorded to

`StreamingDatasetReader.open` -- behind `strands_robots.stream_dataset` and
`Simulation.stream_dataset`, documented as the in-process counterpart to
`start_recording` / `stop_recording` -- forwarded the caller's `root`
unresolved, so the read-back half of that loop looked for the dataset somewhere
the recording had never been. A `repo_id` that is itself a path (no
`owner/name` slash, or `./`-prefixed) is a local directory here and is not one
to LeRobot, which derives any absent root as `$HF_LEROBOT_HOME/{repo_id}`
whatever the id looks like -- and `StreamingLeRobotDataset` reads a local
dataset only when it is given a root, so the miss fell through to a Hub lookup
for a name that only ever meant a directory. Measured on lerobot 0.6.2:
`start_recording(repo_id="sim_recording", root=None)` plus a 45-step rollout
wrote 7 files under `./sim_recording` with every call `status="success"`, and
`stream_dataset("sim_recording")` then raised `RepositoryNotFoundError`, "404
... https://huggingface.co/api/datasets/sim_recording/refs". The read now
resolves the same rule recording writes through (`local_dataset_dir`) and
streams 45 frames from that directory with no `root` restated. Only that rule:
an `owner/name` id keeps its absent root, which is how LeRobot selects the
revision-safe Hub metadata cache and is already the directory a local recording
under that id wrote to (#3346).
