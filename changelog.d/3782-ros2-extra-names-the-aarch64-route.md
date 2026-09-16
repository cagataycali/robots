### Fixed: every `[ros2]` remedy names the Linux aarch64 route instead of promising a wheel

`pip install 'strands-robots[ros2]'` was documented as "a self-contained wheel"
that "works on Jetson". No cyclonedds release publishes a Linux aarch64 wheel,
so on a Jetson, a Thor dev kit or a robot's onboard PC the command builds the
sdist against a Cyclone DDS C install. The install hint, the rclpy refusal that
offers the RTPS transport as its alternative, the ROS 2 / RTPS integration
pages, troubleshooting, the two RTPS examples and the manifest comment now scope
the promise and carry the recipe (`CYCLONEDDS_HOME` from a sourced distro or a
source build), including when that variable is needed at runtime.

The cell grading the rclpy refusal establishes the missing import with
`tests._blocked_module.blocked` instead of reading the host, so it holds on a
machine with a ROS 2 distro sourced (where it previously skipped) and after a
sibling suite has left a fake `rclpy` in `require_optional`'s memo (where it
previously reported no refusal at all). The grader that forbids assuming the
absence now derives the functions that probe `rclpy`, not only the classes, so a
cell reaching the probe by calling the guard directly is covered too.

