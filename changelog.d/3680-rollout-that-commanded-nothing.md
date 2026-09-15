### Fixed: a rollout that never commanded the robot is an error, not a completed one

`PolicyRunner.run` already refuses one route to an uncommanded rollout: when
every step emits keys and NONE of them resolve to an actuator, the result is
`status="error"` naming the unresolved keys and the robot's valid ones. The
mirror - every step emits no key at all - reached the same physical outcome and
was reported `status="success"` with the text `Policy complete`, because a
policy that names nothing produces no unresolved key to count.

Every field a caller would gate on read healthy: `action_errors` is `0` because
nothing was refused, and `action_resolution_rate` / `partial_action_failure_rate`
are keyed on the robot's actuators, so a robot declaring none contributes an
empty map and a `0.0` rate - which is what a robot with no resolution problem
looks like too. `action_commands_robot`, the rule this module already states for
"an action that commands nothing", was read only by the two evaluation routes,
and its own docstring named `run`'s `action_resolution_rate` as the way this
surface separated the two cases.

Both routes are reachable from shipped code. A model declaring no `<actuator>`
block binds a policy to an empty action-key list, so every action it emits is
empty - the shipped `talos` (45 joints) and `asimov_v0` (15 joints) descriptions
both compile that way, and `robot_action_keys` reports zero keys for each. On a
fully actuated robot, a decode whose rows carry no joint value emits the same
empty action, as `CuroboPolicy._next_chunk` does per waypoint when the planner's
trajectory rows carry no joint position.

`run` now reads `action_commands_robot` too, reports the tally as
`actions_applied` in the rollout payload, and refuses the aggregate beside the
total-unresolved refusal it mirrors, naming which route it was: a robot with no
actuator is told so, and a robot with actuators is pointed at its policy's
decode. `actions_applied` counts actions that commanded at least one key rather
than completed `send_action` calls, since an action naming no key reaches the
backend like any other. The per-step tolerance is unchanged - a single empty
action among commanded ones is legitimate policy behaviour - and partial
under-actuation keeps its documented posture of a `success` carrying
`partial_action_failure_rate`.
