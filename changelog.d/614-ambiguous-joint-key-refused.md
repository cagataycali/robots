### Fixed: dict-form joint writes refuse an ambiguous bare joint key

In a scene with several robots carrying the same joint name or label (two
so101s both have `1` / `shoulder_pan`), `set_joint_positions` and
`set_joint_velocities` in dict form with no `robot_name` wrote the first
robot attached and reported success, while the list form, `get_robot_state`,
`move_to` and `run_policy` on the same scene refused to guess. The dict form
now refuses too, naming the robots that carry the key and both remedies
(`robot_name=` or a qualified key). A bare name only one robot carries still
resolves.
