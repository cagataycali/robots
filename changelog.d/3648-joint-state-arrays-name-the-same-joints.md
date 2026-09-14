### Fixed: a published `JointState` names the joints its positions belong to

`SimEngine._publish_ros_telemetry` built `JointState.name` from
`robot_joint_names()` but `JointState.position` from a filtered read of the
observation. The two arrays are one table paired by index, so filtering one
column compacted it: any joint the observation did not carry shifted every later
joint onto its neighbour's value, and the tail fell off the end unreported.

Every floating-base robot reached this on every step - the root freejoint is
joint 0 of `robot_joint_names()` and is never an observation key - which is 16
of the 61 robots in the registry that load in MuJoCo. Measured on Spot, 18 of
its 20 joints were published under the wrong joint's angle, off by up to
0.256 rad (14.7 deg), and `arm_f1x` was never published at all. The message was
well-formed, DDS delivered it, and no subscriber could tell.

The pair is now built together, so a joint without an observation drops its name
as well as its value. `publish_joint_states` also grades the two arrays on both
transports and drops a mismatched pair whole with a warning naming both counts,
for the reason `_command_action` refuses a malformed inbound command whole
rather than applying part of it: a partial state is not a state.
