### Fixed: `set_joint_positions` refuses a pose outside a limited joint's range

`mj_forward` does not clamp `qpos`, so a value past a joint's `range` was
written and reported as `Set n/n joint positions, FK updated`; the next step
then drove the joint back through its limit at whatever velocity the constraint
solver produced (99 rad on the so100 `[-1.92, 1.92]` base joint left it at
-9.4 rad moving 23.8 rad/s after 100 steps). The call now refuses before any
write, naming each joint, value and range; a joint without a `range` still
accepts any finite value.
