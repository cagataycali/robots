### Fixed: `HardwareRtpsBridge` publishes each robot's `joint_states` on that robot's own topic

`publish_joint_states` takes `robot` per call and derives
`/<robot>/joint_states` from it, but the pure-RTPS bridge cached a single
DataWriter for the whole bridge. The first robot to publish owned the only
writer, so every later robot's `JointState` went to that robot's topic: a
subscriber to the second robot's `joint_states` found no writer at all, while
the first robot's topic carried both arms interleaved - and the sample's own
`frame_id` still named the robot it came from, so nothing reported the
mismatch. The joint writers are now cached per robot, matching the
`_image_writers` cache in the same class and the rclpy transport's per-robot
publisher cache, which is what makes the two transports advertise the same
topics for the same calls. `shutdown()` drops every robot's writer.
