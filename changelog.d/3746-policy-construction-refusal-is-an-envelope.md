### Fixed: a policy the provider's constructor refuses is reported, not raised or lost

`run_policy`, `eval_policy` and `evaluate_benchmark` now return the
`status=error` envelope every other refusal returns when the provider's
constructor rejects its `policy_config` (an invalid port, a keyword the
constructor does not bind) instead of raising past it. A rollout whose
constructor fails on `start_policy`'s worker - previously reported as started
and then as "No policies running.", with nothing logged - is logged at ERROR
and listed by `list_policies_running` under "Rollouts started with
start_policy that ended in error", until that robot's next rollout.
