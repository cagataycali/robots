### Fixed: a stop pressed after `start_task` returned and before the rollout began is honored

`Robot._execute_task_async` cleared the stop latch as its first action, so a
`stop_task()` landing between `_claim_task` and the executor picking the job
up was erased: the stop was answered "No task running to stop" and the arm then
ran the whole rollout. The latch is now cleared by `_claim_task` under the
admission lock, the claim records the task as `connecting`, and the rollout
checks the latch before connecting - the task ends `stopped` with zero commands
sent.
