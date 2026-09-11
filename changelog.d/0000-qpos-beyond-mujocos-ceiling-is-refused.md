### Fixed: a pose too large to be a joint coordinate is refused instead of resetting the world

`mj_step` runs `mj_checkPos` before it integrates, so a `qpos` entry past
`mjMAXVAL` (1e10) makes MuJoCo declare the simulation unstable and reset the
ENTIRE state - every joint of every robot and every object, back to its initial
value - reporting that only as a `WARNING` line on stderr. Three MuJoCo surfaces
wrote a caller-supplied value straight into `qpos` without holding it to that
ceiling, so each reported success over a world the next `step` destroyed.

A joint's declared range already bounded a limited joint, which left the joint
that declares none: on g1, `set_joint_positions({"g1/floating_base_joint":
1e11})` returned `Set 1/1 joint positions, FK updated`, and one step later
`data.time` had rewound 0.802 -> 0.002, a settled cube was back at its 0.45 m
spawn and a shoulder holding 1.094 rad read 0. `move_object` and `add_object`
were the other two: a dynamic body's freejoint pose lands in `qpos` verbatim,
including its quaternion, which is not renormalized on write.

All three now refuse before writing anything, through one shared check, naming
the component and the reset it avoids - the stance `set_joint_velocities` has
taken on `qvel` since D-052. The ceiling is MuJoCo's own, so `mjMAXVAL` exactly
is still writable, a non-unit quaternion is still a usable direction, and a
static body - welded with no freejoint, owning no `qpos` entry, stable however
far away it sits - is not held to the ceiling at all.
