### Fixed: an Isaac Lab run says why it failed, whether it solved its task, and can be stopped

`train_policy(action="status", provider="isaaclab")` read only the mean reward
and the last twelve log lines. A run whose reward became `nan` kept reporting
its last finite reward as the latest one; a crash put the traceback far above
the tail, because the child's stdout was block-buffered, so NaN observations,
CUDA out-of-memory and an unknown task all read "Isaac Lab exited 1" followed
by iteration metrics; and a curriculum task whose penalties ramp up looked like
it was not learning while its own `Metrics/success_rate` said 0.98.

Status now reads `nan`/`inf` rewards (`diverged`, `diverged_at_iteration`),
counts with thousands separators, every `Metrics/`, `Curriculum/` and
`Episode_Termination/` term (`task_metrics`, `success_rate`), and judges
`learning` on ten-iteration windows (`reward_trend`, `best_iteration`). A failed
run names its cause first - `metrics["failure"]` is `cuda_oom`,
`nan_observation`, `diverged`, `unknown_task`, `unknown_physics_preset`,
`timeout`, `killed`, `exception` or `exit_status`, with the exception line in
`metrics["error"]` and a next step in the message - and the child runs with
`PYTHONUNBUFFERED=1` so the log keeps its order.

`train_policy(action="stop", job_id=...)` stops a running job and reports
`stopped` with its checkpoints kept (`Trainer.stop`, which a trainer without
runs in flight refuses). A second launch of the same task into the same
`output_dir` while the first is alive is refused with the running `job_id`.
`validate` refuses a task id the Isaac Lab install does not register, with the
closest matches, read from its task packages without starting Isaac Lab. Each
run writes `strands_run.json` (task, physics preset, environments, seed,
overrides) beside its checkpoints.
