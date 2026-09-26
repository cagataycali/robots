### Docs: the pure-RTPS indistinguishability claim names the graph metadata a ROS 2 node still sees

Both hardware bridges publish the same topics and a real `rclpy` subscriber
decodes their `JointState` field for field, but a bare DDS participant carries no
ROS 2 node name and no type hash: `ros2 node list` does not list it, `ros2 topic
info -v` reports its publisher as `_CREATED_BY_BARE_DDS_APP_` with an `INVALID`
type hash, and each `rmw_cyclonedds_cpp` subscriber logs one "Failed to parse
type hash" warning. The six surfaces that promised a ROS 2 observer could not
tell the transports apart now scope that to the payload and point at
`docs/ros2/rtps-robot.md`, which names every axis the graph exposes - including
the `KEEP_LAST (1)` writer history where the rclpy bridge takes `qos_depth=10`.
