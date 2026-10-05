### Fixed: run_multi_policy refuses a policies value that is not a Policy

`run_multi_policy(policies={robot: value})` now returns a structured error naming
`policies['<robot>']` when a value is a provider name, a config dict, `None`, a
scalar or a `Policy` class instead of an instance. It used to raise a bare
`AttributeError: ... has no attribute 'get_actions'` on the first step (MuJoCo
and Isaac). The message is the one `run_policy(policy_object=...)` already gives.
