### Fixed: `stop_policy` without a robot names the robots whose rollouts are running

`robot_name` stays required on `stop_policy`, but `start_policy` on the sole
robot defaults to it, so an agent that launched without a name met a dead-end
"requires 'robot_name'". The refusal now names the running robots with the
exact call (or says nothing is running); backends without a rollout registry
keep the bare requirement.
