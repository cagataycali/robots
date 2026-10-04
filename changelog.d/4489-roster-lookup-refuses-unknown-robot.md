### Fixed: `robot_joint_names` and `robot_action_keys` refuse a robot the scene does not hold

Both lookups returned `[]` for a name that was not in the scene, so a robot added as `Robot("g1")` and asked for as `robot_joint_names("unitree_g1")` handed back an empty roster that bound a policy's state keys to nothing. They now raise `ValueError` naming the robots that are there, on MuJoCo, Newton, Isaac and mjlab alike (mjlab raised `KeyError` before). Code that relied on `[]` for an absent robot should check `list_robots()` first.
