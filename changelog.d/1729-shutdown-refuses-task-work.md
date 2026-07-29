### Fixed: a shut-down hardware `Robot` no longer reports a rollout it never drove

`cleanup()` (and `stop()`, which calls it) sets a shutdown latch and never clears
it, then releases the task executor, the mesh and the ROS bridges. The control
loop's condition honored that latch, but nothing else on the task path did, so it
produced two misreports.

A task started after `cleanup()` was admitted, took the motors-bus claim,
commanded the arm zero times, and came back
`Policy rollout completed: 0 steps in 0.0s` with `status="success"`.
`start_task` was worse: its submit reached the already-shut-down executor and
surfaced a bare `RuntimeError: cannot schedule new futures after shutdown` past a
method whose contract is a tool-shaped result. `run_policy`, `start_task` and the
`execute` action now refuse a shut-down robot by name, before the bus claim, so
the refusal cannot leave the robot rejecting later work either.

The second misreport could strike a rollout that really was driving the arm.
`cleanup()` sets the latch first and only then calls `stop_task()` for a rollout
it finds `RUNNING`; a loop that has already exited on the latch has left
`RUNNING`, so `stop_task()` is never called and the stop latch stays clear. The
terminal block consulted only the stop latch, so a rollout truncated two
hundredths of a second into a thirty-second budget reported
`completed: 24 steps`. It now consults both latches its own loop condition
honors and reports `STOPPED`. A rollout that reaches its own budget is still
`COMPLETED`.
