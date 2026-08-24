### Fixed: a hardware task step cap the control loop cannot count against is refused

`Robot.run_policy(n_steps=...)` and the `_execute_task_sync` chokepoint now
validate the optional step cap against the shared positive-count domain, the
same rule the loop's `action_horizon` and the simulation's `run_policy` step
horizon already apply. The loop bounds a rollout with
`n_steps is None or step_count < n_steps`, ANDed with the `duration` budget, and
the cap beside `duration` in the same signature was read straight into that
comparison: `0`, a negative, `nan` and `False` made it false on its first
evaluation, so the task reported `status="success"` and "Policy rollout
completed: 0 steps" for a rollout that never queried the policy and never
commanded a servo; `inf` was never false, so the requested cap silently
vanished and the rollout ran to the `duration` budget instead; `True` read as a
silent cap of one and `2.7` stopped after three applied actions, a count the
caller never named; and a non-numeric cap surfaced a bare `TypeError` naming a
comparison internal rather than the parameter. `None` remains the documented
"no cap" spelling. The refusal lands before the policy is initialized and before
the bus is claimed, so a cap that cannot be honored costs no inference and no
servo write.
