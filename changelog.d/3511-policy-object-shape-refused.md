### Fixed: a `policy_object` the rollout cannot drive is refused at the entry point

`policy_object` hands a rollout a pre-built policy, so it bypasses provider
resolution and the provider's `preflight` hook - the two checks that would
otherwise have something to say about it. Its own shape was never checked, so
`run_policy(policy_object=42)` raised a bare `AttributeError: 'int' object has
no attribute 'set_control_frequency'` (and `eval_policy` the same for
`set_robot_state_keys`), naming a library internal rather than the parameter the
caller got wrong; passing the policy CLASS instead of an instance surfaced as
`TypeError: 'property' object is not iterable` from inside a tree walk. On
`start_policy` that raise happened on the executor worker, whose result nothing
reads, so the caller was handed `status="success"` / "Policy started" for a
rollout that applied no action, after which `list_policies_running` reported
nothing running - the same reading a completed rollout gives.

`policy_object` is now held to its domain at the public entry point, beside the
two keyword bags (`policy_config` / `policy_kwargs`) that were already checked
there for this reason: `run_policy`, `start_policy`, `eval_policy` and
`evaluate_benchmark` each return a structured error naming the parameter, what
arrived, and how to get a usable value, and a `Policy` subclass is told to
instantiate itself. The refusal precedes the robot claim, so a rejected
`start_policy` leaves the robot startable. A `Policy` instance and an omitted
`policy_object` are unaffected.
