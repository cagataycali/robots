### Fixed: a gated command with no operator to ask says what to set

A robot command stopped by the operator gate with nobody reachable now names
the setting that pre-approves it - `STRANDS_ROBOT_COMMAND_ALLOW=execute`,
`STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`, or `<variable>=*` for every command of
that tool - instead of the variable alone. When the variable is set to a value
that pre-approves nothing (`1`, `true`, a typo) the refusal says so, so the
same message no longer comes back after the advice was followed.

Both ways of having no operator carry that remedy: no `tool_context` to raise
an interrupt through, and a host that has one and refuses to interrupt, which
previously named the reason and no remedy at all. The interrupt's resume line
and these refusals take the value from the same matcher, so they agree. A
command the operator declined is unchanged - that question was answered.
