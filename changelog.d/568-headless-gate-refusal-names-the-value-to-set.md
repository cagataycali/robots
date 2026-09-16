### Fixed: the headless operator-gate refusal says what to set

A robot command refused because no operator was reachable now names the
setting that pre-approves it - `STRANDS_ROBOT_COMMAND_ALLOW=execute`,
`STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`, or `<variable>=*` for every command
of that tool - instead of the variable alone. When the variable is set to a
value that pre-approves nothing (`1`, `true`, a typo) the refusal says so,
so the same message no longer comes back after the advice was followed. The
interrupt's resume line and this refusal carry the same spelling.
