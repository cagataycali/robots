### Fixed: a non-string `instruction` is refused by the single-robot rollout verbs

`run_policy`, `start_policy`, `eval_policy` and `evaluate_benchmark` accepted
`instruction=42`, `None`, a list or a dict, ran the rollout and wrote the value
into the summary, the result metadata and the recorded dataset's `task` column,
while `run_multi_policy` already refused the same value. They now return
`status="error"` (`"<verb>: 'instruction' must be a string (use \"\" for none),
got <type>."`) before a policy is built or the robot is claimed. `""` stays the
default.
