### Fixed: `start_policy` refuses the policy configurations `run_policy` refuses

`start_policy` submits its rollout to a worker thread, and nothing reads that
worker's result - the `Future` is tracked only to answer "is this robot busy"
and is pruned once done. Every other knob on the surface is therefore checked
before the submit (horizon, duration, seed, video config, provider keyword
bags, recording rate); the policy configuration was checked only half-way. The
provider *name* was resolved synchronously, but the provider's own class-level
`Policy.preflight` hook - the check that refuses a camera the model's declared
image inputs cannot be routed from, or an action-chunk count the consumer
cannot execute - ran on the worker, where its refusal was discarded. A
`policy_config` that `run_policy` rejects up front with a message naming the
parameter was reported as `status="success"`, "Policy started on '<robot>'
(async)", and a moment later `list_policies_running` said "No policies
running." - the same reading a completed rollout gives.

Both surfaces now give one verdict: `start_policy` runs the same
`_preflight_policy_config` as `run_policy` before claiming the robot, so the
refusal is returned verbatim to the caller and the robot is left unclaimed. As
on the blocking surface, the check is skipped for a pre-built `policy_object`
(the provider is then unused), which also removes a false refusal - an
unresolvable `policy_provider` used to reject a `policy_object` rollout that
`run_policy` accepts. It stays free for providers that leave `preflight` alone:
the observation that feeds the hook, which renders every camera in the scene,
is still gathered only when a real hook will read it.
