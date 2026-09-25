### Docs: the pure-RTPS page keeps the tool, acting as a robot gets its own page

`docs/rtps-integration.md` documented two subjects in one 2,154-word scroll: the
`use_rtps` tool (install, actions, IDL bundle, the topic-name rule) and how a
robot appears on a ROS 2 graph over RTPS (`RtpsRobot`, the rclpy-free hardware
bridge, and the DDS Security gate on the inbound `joint_command` surface). The
second half moves to `docs/ros2/rtps-robot.md`, leaving 1,276 and 990 words, so
neither page owes the 1,500-word budget an exemption. Inbound anchors from
`docs/ros2/hardware-bridge.md` and `docs/security/hardware.md` follow the section
they name. No fact, refusal string or fence is reworded; the rationale passages
the changelog already carries are deleted rather than moved.
