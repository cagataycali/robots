### Added: a rollout that stalls against contact says so in `saturated_step_rate`

Every action-health field `run_policy` reports is keyed on key resolution:
whether an action name reached an actuator. An arm pressing its gripper into the
floor resolves every key on every step, so it read `status="success"`,
`action_errors=0` and `partial_action_failure_rate=0.0`, the same as a rollout
that reached its command.

The payload now carries `saturated_step_rate` (the fraction of steps on which
any of the robot's actuators sat at its force limit) and `saturation_rate` (the
same per actuator), and the text names the pinned actuators. They come from the
new `SimEngine.saturated_actuators(robot_name)`; MuJoCo reads
`actuator_force` against `actuator_forcerange`, and a backend that cannot tell
reports `None` for both fields.
