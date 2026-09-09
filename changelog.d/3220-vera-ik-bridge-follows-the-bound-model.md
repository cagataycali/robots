### Fixed: the VERA IK bridge a rollout solves on is a bridge for the model that is bound

`VeraPolicy._ensure_ik_bridge` cached the `MinkIKBridge` in a single slot keyed
on nothing, while the bridge holds a `mink.Configuration` and both tasks built
from ONE compiled `MjModel`. The simulation rebinds a policy on every rollout
(`bind_policy_sim_context` -> `set_sim_context`) and hands it the model compiled
*now*, which is a new object after any scene change; `autoconfigure_ik` returns
early once an ee-frame is configured, so the rebind never re-entered
`set_ik_target` and nothing invalidated the bridge.

Both outcomes were silent at the boundary. When the DOF count changed, the stale
bridge refused the seed built from the bound model - `solve: 'q_init' must be a
9-element vector, got 16` - naming the caller's seed and the superseded model's
`nq`. When it did not change (a robot re-placed, a scene rebuilt), the solve
returned clean-looking joint targets for geometry that is no longer there.

The cache is now keyed on every input the build reads: the model by identity,
the end-effector frame name, and the frame type - the same shape the adjacent
`_joint_qpos_addr` cache already uses for the same reason. An unchanged model and
frame are still served from the cache.
