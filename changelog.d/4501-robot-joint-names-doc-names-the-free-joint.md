### Docs: `SimEngine.robot_joint_names` says its roster includes a floating base's free joint

The docstring said this list orders the `observation.state` columns a recording
writes, so a policy could be keyed by it. On a floating-base robot (`g1`,
`unitree_h1_2`, `spot`) the list starts with the 6-DoF free joint, which has no
scalar column, so it is one wider than that vector. It now says so and points
policy binding at `robot_action_keys`. No behaviour change.
