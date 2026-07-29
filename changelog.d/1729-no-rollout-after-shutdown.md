### Fixed: a rollout on a shut-down robot is refused instead of reported as completed

`_shutdown_event` is one of the hardware control loop's exit conditions, but it
was neither an admission check nor a terminal-status discriminator. Once
`cleanup()` had set it, a task ran the whole bring-up and then fell out of the
loop on its first evaluation, so `run_policy` and the agent-tool `execute`
action both returned `status="success"` with `steps: 0` - indistinguishable from
a rollout that really drove the arm - while `start_task` instead raised
`RuntimeError("cannot schedule new futures after shutdown")` from the executor
submit, naming a `concurrent.futures` internal rather than the robot. Bring-up
is not side-effect-free: it re-opened the motors bus and warmed every camera,
and because `cleanup()` does not disconnect the robot and the executor is
already down, those devices stayed open for the life of the process; it also
called `Policy.reset()`, clearing the per-episode state of a policy object the
caller may still be driving. All three entry points now refuse in the same tool
shape before any device is touched. Separately, a `cleanup()` landing while a
task is still connecting sets no stop latch - `cleanup()` calls `stop_task()`
only for a `RUNNING` task - so the terminal block now treats `_shutdown_event`
the way it already treats the stop latch and reports such a rollout `stopped`
rather than `completed`.
