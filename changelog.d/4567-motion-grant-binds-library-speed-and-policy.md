### Fixed: a motion approval binds the pose library, speed profile and policy it was given for

A dashboard yes for a `pose_tool` or `serial_tool` motion, or for a robot's
`execute` / `start`, was filed under a key that left out `pose_tool`'s
`robot_id` (the pose library a `pose_name` is read from), its `smooth` /
`step_delay` speed profile (and `steps` when it equalled 20), and the policy
fields (`policy_provider`, `policy_host`, `policy_port`,
`pretrained_name_or_path`, `policy_type`, `embodiment`, `model_path`, `walk`,
`target_velocity`). A yes for one library's pose, a slow interpolated move, or
the `mock` policy was spendable by another library's pose, a full-speed write,
or a checkpoint. Each of those fields is now keyed and shown to the operator,
an omitted default keys the same as the value the tool runs with,
`pose_tool` matches the grant against the calibration records it builds its
controller from, and a call it then refuses on its inputs spends its grant.
