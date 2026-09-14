### Fixed: a refused `HardwareRosBridge` no longer strands the rclpy context

`HardwareRosBridge.__init__` validated `joint_limits` *after*
`super().__init__()`, which writes the process-wide `ROS_DOMAIN_ID`, calls
`rclpy.init()` when nothing else has, and creates the node. A malformed bound
therefore raised having already re-pointed the process at the requested domain
and left an undestroyed node behind - and because `__init__` raised, the caller
holds no bridge, so the `shutdown()` that releases the context and destroys the
node is unreachable. A corrected retry cannot clean up either: it reads
`_owns_context` from `rclpy.ok()`, finds the context already up, declines
ownership, and shuts down only its own node.

The mapping is now validated alongside `spin_period` and `enable_commands`,
which the same constructor already ordered ahead of the base for exactly this
reason ("leaves the environment as it found it"), and which the pure-RTPS
sibling `HardwareRtpsBridge` already applies to the same mapping before it
builds its participant. Refusing a bridge now costs the process nothing: no
`ROS_DOMAIN_ID` write, no rclpy context, no node.
