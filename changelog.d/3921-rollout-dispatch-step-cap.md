### Fixed: a rollout dispatch pin counts a step cap, not the actions a 0.2 s window fits

`TestTheRolloutIsDrivenExactlyOnce` asserted ten applied actions where the
rollout's only horizon was `duration=0.2` at 50 Hz, so the pinned count was
whatever the host fit into 0.2 s - with 16 busy cores beside the run the two
cells reported 8 and 9 applied actions and named the dispatch for the load.
`_run_control_loop` already accepts `n_steps`, ANDed with the duration in the
loop condition, so the pin caps the rollout at ten steps and keeps `duration`
as a ceiling it reaches first by two orders of magnitude. The count stays
exact and stays falsifiable: `_execute_task_async` zeroes `step_count` per
task, so a rollout driven twice still reports twenty.
