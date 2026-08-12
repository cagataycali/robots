### Fixed: a non-finite `joint_limits` bound is refused at bridge construction

`RosTelemetryBase._validate_joint_limits` exists so a malformed joint bound
refuses the bridge at build time rather than becoming a silent mid-run rejection
of every command. Its only ordering check was `low > high`, and every comparison
against `nan` is False, so a `nan` bound passed validation and then made the
per-command `low <= pos <= high` check False for every position: the bridge
accepted construction and silently dropped **every** inbound `joint_command` for
that joint. An infinite bound passed the same way and can never constrain
anything. Both bounds now go through the shared `finite_number_error` domain
before the ordering comparison, matching the finiteness check
`IsaacDeltaEEFController` already applies to its own `joint_limits`. Both
hardware bridges inherit the validator, so one guard covers ROS 2 and pure-RTPS.
To leave a joint unconstrained, omit it from the mapping.
