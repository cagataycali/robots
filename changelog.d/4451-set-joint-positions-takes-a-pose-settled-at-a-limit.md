### Fixed: `set_joint_positions` writes back a pose the simulator settled at a joint limit

Joint limits are soft, so a servo held against one rests slightly outside the
range, and `get_robot_state` reports that. The range guard compared against the
bare range and refused the whole pose: an SO-101 resting on its lower limits read
back wrist flex 1.1e-3 rad and wrist roll 1.9e-4 rad past them, and writing that
pose back failed with "outside the joint's range, nothing written". MuJoCo, Isaac
and Newton now share one rule, `outside_joint_range`, which allows
`JOINT_RANGE_WRITE_TOLERANCE` (0.01 rad, at most 1% of the range) of slack. A
value inside the band is written as given, never clamped. Anything past it is
still refused.
