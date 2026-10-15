### Added: every run can be a LeRobot v3 dataset

`robot.run(policy, record="data/pick")` appends the run as one episode of a LeRobot v3 dataset: state and action
in the policy's units (declared in `meta/units.json`), camera frames as video, the instruction as the task. A second
run, in any process, resumes the same dataset; a different fps or robot is refused. A run that raised or faulted is
dropped; any other ending (steps, `stop()`, an e-stop) saves it. `record="hf://owner/name"` writes into LeRobot's
cache and uploads only with `push=True`. Recording needs LeRobot's dataset extra; the run loop calls
`strands_robots.recording`, never the reverse.
