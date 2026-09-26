### Fixed: a cmd_vel demo raises the refusal the command gate returned

`/turtle1/cmd_vel` is a gated command surface and a script has no
`tool_context` for an operator, so `drive()` reports a refusal by RETURNING
`{"status": "error", ...}` naming `STRANDS_ROS2_COMMAND_ALLOW` /
`BYPASS_TOOL_CONSENT`. `examples/ros2/rtps_turtle_demo.py` discarded that result
and printed `done - the turtle should have moved` against a real `turtlesim` that
had not moved at all; `examples/ros2/turtlebot_demo.py` printed an `after:` pose
identical to its `before:` one. Both now check the envelope, and the RTPS demo's
header documents the pre-approval its rclpy twin already did.
