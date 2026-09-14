### Fixed: a hardware bridge's `command_robot_name` must be able to name a topic

Both hardware ROS 2 bridges let a caller override the namespace their inbound
`joint_command` topic is read from. It is the one caller-supplied value the
module renders into a topic through its name sanitiser - every other name
reaching it is derived internally - and it was ungraded, so both non-string
outcomes were worse than a refusal.

A truthy non-string raised out of the sanitiser's `re.sub`
(`TypeError: expected string or bytes-like object, got 'int'`), naming no
parameter, from *after* the transport was built: on `HardwareRosBridge` the
process-wide `ROS_DOMAIN_ID` had been rewritten, the rclpy context started and
the node created; on `HardwareRtpsBridge` the `DomainParticipant` existed.
`__init__` raising returns no object, so the `shutdown()` that releases them is
unreachable. A falsy non-string (`0`, `[]`, `False`) did not raise at all: the
`command_robot_name or <derived>` default filtered it out and the bridge
reported success reading commands under the bound robot's own name, a namespace
the caller never asked for.

Both bridges now grade it where they already grade `domain_id`, the poll/spin
period and `enable_commands` - ahead of the transport - so the refusal names the
parameter, states the input that restores the default, and costs the process
nothing. `None` and `""` still select the bound robot's name.
