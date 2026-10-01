### Fixed: a pre-built `policy_object` keeps the robot keys its caller bound it to

`run_policy`, `eval_policy` and `evaluate_benchmark` called
`policy.set_robot_state_keys(robot_action_keys)` on a caller's `policy_object` too,
overwriting whatever keys it had been configured with on every rollout - so a
pi0.5-DROID policy bound to the panda's 7 arm joints and one finger (8 of its 9
keys) was re-bound to all 9 and could not be driven the way it was trained. A
pre-built policy now keeps its `robot_state_keys` when every one of them is a key
of the robot being driven (an info line names the count); generic `joint_<i>`
placeholders, keys the robot does not have and an unset list are replaced with the
robot's action keys as before, and a policy the call builds itself is bound exactly
as before.
