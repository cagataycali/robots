### Added: `train_policy(action="record")` records an Isaac Lab policy rollout as a LeRobotDataset

Isaac Lab's `play` writes one viewport mp4 and `capture_env_sensors` is train-only, so
every Isaac Lab dataset strands produced needed an out-of-tree harness running strands
inside the Isaac Lab interpreter. `IsaacLabTrainer.record(job_id, dataset_dir, ...)`
(`action="record"`) launches `_isaaclab_record_runner.py`, shipped with strands and run
by `ISAACLAB_PYTHON` without importing strands, which rolls the run's newest rsl_rl
checkpoint out in `episodes` parallel environments with the physics and `env.*`
overrides it trained with, through a fixed camera per env, and writes each episode's
arrays; `status` of the returned job converts them with `DatasetRecorder` into a
LeRobotDataset (`observation.state` = joint positions + policy observation + root pose,
`action` named by the action terms' joints, `observation.images.camera`). On one L40S a
Cartpole run trained for 60 iterations recorded 3 episodes / 360 frames at 60 fps that
`LeRobotDataset` loads, video included.
