### Fixed: a ROS 2 node that will not be destroyed no longer holds the rclpy context

`RosTelemetryBridge.shutdown` releases two independent resources - the bridge's
node handle, and the process-wide rclpy context when it was this bridge that
called `rclpy.init()`. A `destroy_node` failure (rclpy raises `InvalidHandle` or
`RCLError` out of the C layer) propagated out of the first step and skipped the
second, and nothing retried it: the bridge keeps claiming ownership while the
next bridge in the process finds `rclpy.ok()` already true, disclaims ownership,
and never shuts the context down either. Both call sites - `SimEngine.cleanup`
and `Robot.cleanup` - tear the bridge down inside a suppressing block, so the
leaked participant and its discovery threads were reported to no one.

Both releases now run. The node failure is logged at debug, because the context
shutdown takes the node down with it; a context that will not shut down is
logged at *warning*, because that failure is the last word on a participant
still on the wire.
