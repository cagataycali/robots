### Fixed: `URDriver` holds every RTDE read to the axis count its command path enforces

`URDriver` names each vector the controller answers by zipping it against
`JOINT_NAMES`, and `_read_joints` owns the rule that makes that legal: a width
other than six is refused by name, because this driver serves six-axis e-Series
arms only. `send_action` was held to it. `state()` was not - it re-read
`getActualQ` itself and zipped with `strict=False`, so the same controller
`send_action` refused was reported as `status="success"`.

A five-axis answer lost its last joints. A seven-axis answer had its extra
element truncated away, producing six named joints that are shaped exactly like
a genuine six-axis read, so nothing downstream could tell the two apart. Since
`state()` also writes the cache `get_observation` serves - and `connect_eagerly`
primes it through `_absorb_state` - that pose is what reached the mesh joint
read. `joint_velocities` carried the same truncating zip, and a velocity vector
that disagreed with the position vector was reported as far as it went.

The width rule now has one module-level owner, `_axis_count_refusal`, read by
`_read_joints` and by `state()`, so the read and command surfaces cannot come to
disagree about which controllers this driver serves. `state()` asks
`_read_joints` for the joint vector and refuses with its reason, holds the
velocity vector to the same rule, and both zips are `strict=True` once the width
is established. `connect_eagerly` still succeeds and still ignores a refused
priming read; the difference is that it no longer seeds the mesh with a pose the
driver cannot name.
