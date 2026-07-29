### Fixed: a hardware task refuses a duration the control loop cannot honor

`Robot.run_policy` / `start_task` / `execute_task` bounded every rollout by
comparing elapsed wall-clock time against `duration` without validating it, so a
budget the loop could not honor was spent on the arm instead of refused. A
`duration` of `0`, a negative value, or `nan` made the loop condition false on
its first evaluation: the task reported `status="success"` for a rollout that
never queried the policy and never commanded a servo, and `start_task` reported
`Task started` for the same. `inf` never made it false, so the loop commanded
the servo bus indefinitely and the blocking call never returned. A non-numeric
budget reached the comparison intact and surfaced a bare `TypeError` naming a
comparison internal (`'<' not supported between instances of 'float' and
'str'`) rather than the parameter. `True` ran a silent one-second task.

`duration` is now validated at every public entry point, against the same
`positive_finite_number_error` domain as the loop's `control_frequency` and the
simulation's `SimEngine._validate_duration`, so a budget cannot be refused for a
digital twin and accepted for the arm it mirrors. `start_task` checks before
submitting the work, so a budget that cannot be honored is reported to the
caller instead of as a started task, and `_execute_task_sync` checks on its own
because the agent-tool `execute` action and the mesh `execute` dispatch reach it
directly with a peer-supplied value. Unlike the simulation - where an `n_steps`
recomputes `duration` and supersedes it - the hardware loop ANDs the two
conditions, so `duration` is validated even when a step cap is given.
