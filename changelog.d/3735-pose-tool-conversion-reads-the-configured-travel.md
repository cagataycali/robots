### Fixed: `pose_tool` scales a joint off its configured travel, not off its name

`MotorController.degrees_to_position` and `position_to_degrees` selected the
gripper's mapping by name and hard-coded 0-100 in place of that joint's
configured `range`, so the conversion read no bounds at all for one of the six
joints. The comment above `_DEFAULT_MOTOR_CONFIGS` states the invariant that
breaks: `range` is consulted twice, by the conversion and by the guard in
`_joint_target_error`, because "a second copy of these bounds could disagree
with the one the servo is actually driven from".

On the shipped table the two arithmetics coincide - the gripper's default range
is `(0, 100)` - so the special case changed nothing until a caller retuned the
travel, which `motor_configs` is a per-instance copy in order to allow. With
`range` set to `(0, 50)`, a fully-open command wrote 2047 counts, half of the
encoder, while the guard accepted that target against those same bounds; and
`position_to_degrees` read the count back as `50.0`, so the write and the read
shared the wrong scale and the arm reported the value it had been asked for
while sitting at half travel.

Both directions now scale off the configured bounds, one rule for every joint:
the unit is a property of the bounds, not of the name, and the gripper's percent
open is simply its range being 0-100. The shipped mapping is unchanged and
pinned as such.
