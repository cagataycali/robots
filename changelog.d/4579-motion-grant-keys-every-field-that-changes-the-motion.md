### Fixed: an operator's yes covers only the motion it was shown

A dashboard approval was filed under a key that left out fields that change
what the robot does: the pose library `robot_id` selects, the speed profile
(`step_delay`, `smooth`), the bus `baudrate`, and which policy an
`execute` / `start` runs (`policy_provider`, `policy_host`, `policy_port`,
`policy_config`, the checkpoint fields the mesh carries, and
`target_velocity`). A yes for one of those calls could be spent by another
that differed only there. Each is now part of the grant and shown to the
operator; every other parameter a gated tool declares is listed with the
reason it is not (`UNKEYED_FIELDS`). `pose_tool` matches a grant against the
calibration records it loaded rather than a second read of the file, and a
yes for a `pose_tool` / `serial_tool` call that stopped before its gate is
forgotten when that call returns.
