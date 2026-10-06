### Fixed: a retained stop or release is refused by every subscriber, and can be cleared

A robot already refused a retained `strands/safety/estop` or
`strands/safety/resume`. The dashboard did not, and folded a stored stop into
the fleet lockout; a `Mesh.subscribe` reader (the `robot_mesh` tool's
`subscribe`) put it in its inbox as if it were live; a wildcard subscription
such as `strands/safety/**` still asked the broker to replay it; and an MQTT
packet whose retain flag could not be read counted as live. All four now treat
the message as stored and drop it, logging once per topic. The new
`clear_retained_safety_messages` (CLI `strands-robots iot clear-retained-safety`,
a dry run unless `--apply`) deletes retained messages an older policy let a
peer store under `strands/safety/`, with a zero-byte retained publish.
