### Fixed: a sim rollout refuses a robot name or instruction that is not a string

`run_policy`, `start_policy`, `eval_policy` and `evaluate_benchmark` used both
values as given. A non-string `instruction` (`None`, a number, a list) ran to
`status="success"` and was written into the result and the recorded `task`
column, while `run_multi_policy` refused it. A `Policy` passed first - the
hardware `run_policy(policy, ...)` shape - was reported as an unknown robot
named by its `repr`. All four now refuse before anything is driven, and the
`Policy` case names `policy_object=` as the fix.
