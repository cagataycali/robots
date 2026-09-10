### Added
- `run_policy` results carry `ctrl_clamped` ({actuator: count} of commands MuJoCo clamped to its ctrlrange during that rollout) in the json block and a one-line note in the text. Until now a policy in the wrong units (deg or normalized onto a radian actuator) returned `status=success, action_errors=0` with a single warn-once log line.
