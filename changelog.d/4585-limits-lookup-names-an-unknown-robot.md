### Fixed: `actuator_ranges` and `saturated_actuators` refuse a robot the scene does not hold

Asked about a name that is not in the world (an alias such as `"g1"` for a
robot added as `"unitree_g1"`, or a typo), `actuator_ranges` answered `{}` and
`saturated_actuators` answered `None`. Those read as "no limited actuators" and
"cannot tell", so the mistake went unnoticed. `robot_joint_names` and
`robot_action_keys` already raise for such a name. Both now raise the same
`ValueError`, which names the robots in the scene and the alias's canonical
name, on every backend (MuJoCo, mjlab, and the `SimEngine` default that Newton
and Isaac inherit).
