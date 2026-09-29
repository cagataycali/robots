### Added: `set_joints`, a simulation-only mesh action

A sim peer's proxy tool advertised a `sim_call` rail that
`mesh.security.validate_command` refuses, so the agent could not move a joint
on any mesh robot. `set_joints` is now an allowed action: `target_joints`
(the dict `start` already carries) plus an optional `hold`, served through the
simulation's own `set_joint_positions` action on a Simulation peer or its child
robot peer. A hardware peer refuses it; real motion rides `execute` and `start`,
where the operator is asked first. The sim proxy now advertises exactly the
actions the wire carries.
