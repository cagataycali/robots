### Fixed: a dataset's declared task count is graded, not converted to the absent case

A validation split is one `eval_split` FRACTION in lerobot, applied as
`ceil(episodes_in_task * eval_split)` per task, so a global `val_episodes` count
is only expressible on a single-task dataset. `validation_split_error` refuses
the request otherwise -- but both readers of `meta/info.json`'s `total_tasks`
converted the header first (`total if isinstance(total, int) and not
isinstance(total, bool) else 0`), and 0 is the value that guard honors as "no
task count recorded". Every declaration outside a bare `int` therefore arrived as
the absent case: a three-task dataset whose header spelled its count `3.0` or
`"3"` passed the guard written to refuse exactly that dataset, and lerobot then
held out 3 episodes where 2 were asked for.

The declaration now reaches the guard verbatim from both the `lerobot_train` tool
and `LerobotTrainer`, and is graded by `declared_count` -- the one owner every
reader of a LeRobot header count already shares. A header declaring something
which is not a count is a third outcome, refused on its own terms and naming the
value, rather than collapsed into the absent one. `0`, `1` and no header at all
are still honored as single-task.
