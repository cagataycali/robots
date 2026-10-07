### Fixed: a simulated `send_action` of only `vx`/`vy`/`vyaw` names the locomotion policy that walks it

`Robot("microduck").send_action({"vx": 0.1, "vyaw": 0.0})` - the twist the
hardware page teaches - was refused in simulation with a list of 14 joints and
nothing else. When every refused key is a twist component and the robot's
registry entry declares a `locomotion_policy` (microduck: `microduck`, G1:
`wbc`), the refusal now spells `run_policy(robot_name=..., policy_provider=...,
policy_kwargs={'target_velocity': [vx, vy, vyaw]})`. A joint typo keeps the
plain refusal.
