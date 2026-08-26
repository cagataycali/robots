### Fixed: `G1Driver.run_policy` / `stop_task` / `get_task_status` now wire the 500 Hz control loop

Previously the three verbs returned `"not wired yet (issue #358)"` refusal
envelopes. The transport primitive the loop needs is `send_action` (landed in
issue #361 PR-B), so the loop can be wired without the `g1_tools` motion
verbs from #358 - the two concerns stay separable.

`run_policy(policy_object, duration=..., n_steps=...)` spawns a daemon thread
that runs at 500 Hz (matching the SDK's own G1 example cadence), builds a
`LowCmd_` from the policy's per-step return, and publishes it on `rt/lowcmd`
through the same `DDSPublisher` `send_action` uses. Every step re-gates
through `_check_motion_gates("motion")` so an FSM transition out of
`HANDSHAKE_FSMS ∪ WALK_FSMS` refuses the *step* rather than the whole task -
the loop exits, publishes a zero-torque `LowCmd_` (from
`_build_zero_torque_lowcmd`, a soft *controlled* stop with Enable on the
named joints), and drops its reference. Same posture on every terminal path
except `publish` (where the wire has just refused; a second stamp would
clobber the reason with a fresh error).

Every exit path names itself in `get_task_status`: `stop_task`, `n_steps`,
`duration`, `gate` (with the refusal text), `policy` (with the exception or
the reason the action was unusable), `publish`. `run_policy` refuses a second
concurrent rollout so two threads cannot silently share `rt/lowcmd`.

`start_task` still refuses because the provider registry that turns an
`instruction: str + policy_provider: "groot"` into a policy object lives in
`strands_robots.policies` and threads through the `g1_tools` motion verbs
(#358). The refusal names both facts and points at `run_policy` as the
already-live path.

Contract-graded off the driver: `tests/drivers/test_g1_control_loop.py`
exercises every exit branch with a callable-double policy and a recording
publisher, so the loop is graded without `unitree_sdk2py` and without a DDS
bus. Every SDK import remains inside a function body.
