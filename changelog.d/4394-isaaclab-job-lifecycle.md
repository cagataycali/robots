### Fixed: an Isaac Lab job stops at its `timeout_s` with nobody polling, a relative jobs dir works, and export takes the asked-for task

Three lifecycle faults of the `isaaclab` trainer. `extra['timeout_s']` was checked
only inside `status()`, so an agent that died after launching a `timeout_s=30` run
left it training and holding the GPU until someone polled; the launch wrapper now
runs a watchdog in the run's process group that writes the `timed_out` marker,
sends SIGTERM to the whole group at the deadline and SIGKILL 15 s later, and leaves
within a second of a run that ends on its own (`play()` gets the same). A relative
jobs dir (`jobs_dir=` or `$STRANDS_ISAACLAB_JOBS`) was passed to a wrapper running
in the output_dir, where it named a directory that did not exist, so every run
ended `error` / "killed"; it is resolved to an absolute path, and so is the exit
file. `export` took the newest run of any task in `output_dir` by modification time,
so "export Isaac-Cartpole" returned the Isaac-Cartpole-Camera policy with
`success`; `latest_checkpoint(output_dir, task=)` filters by the task each run
recorded (`strands_run.json`, else the job record its folder is named after) and
orders by the run's start time, and `export` exports the newest run of
`extra['task']` or refuses when there is none.
