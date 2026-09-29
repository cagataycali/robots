### Fixed: the report of a start_policy rollout can be read after it ends

`start_policy` returned "Policy started" and nothing else, and once the worker finished `stop_policy` answered "Was not running" with no report, so the `run_policy` envelope of every background rollout was lost. The MuJoCo engine now keeps the last completed envelope per robot: `SimEngine.policy_result(robot_name)` returns a copy of it (`None` while in flight) and the first `stop_policy` after the rollout ends on its own carries it as `last_result` (a stop that halted it answers as before).
