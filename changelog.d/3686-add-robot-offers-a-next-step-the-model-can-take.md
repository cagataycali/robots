### Fixed: `add_robot` offers a next step the robot it added can actually take

The `add_robot` summary always closed with `Run policy: action='run_policy',
robot_name=...`. On a model that compiles with no actuators that invitation is a
dead end: `send_action` has no key to resolve, so a policy bound to the robot can
only ever emit an empty action. Both surfaces that consume one already refuse
exactly that condition -- the all-unresolved abort in `PolicyRunner.run` and
`uncommanded_eval_error` for `eval_policy` -- but each reaches its verdict only
AFTER a whole rollout has been paid for. The door that hands out the advice knew
the actuator count one line earlier and offered the invitation anyway, so the
summary for a robot nothing can drive was byte-identical to one for a seven-servo
arm.

This is reachable from shipped code: the registry entry `asimov_v0` resolves to a
pack whose only document declares no `<actuator>`, so `add_robot` reports
`Actuators: 0`, `robot_action_keys` returns `[]`, and the next line invited
`run_policy`. A bare URDF arm loads the same way by construction, URDF having no
actuator concept.

Such a model is still worth adding -- rendering, IK and forward kinematics need no
actuator -- so it is not refused. Instead the closing line now names the step it
can take: `actuate_robot`, the in-tree remedy that adds a position servo per
hinge/slide joint and makes `run_policy` the correct next call. The line is
derived from the robot's resolved actuator ownership, the same count the
`Actuators:` line above it reports, so the two cannot disagree. An actuated
robot's invitation is unchanged.
