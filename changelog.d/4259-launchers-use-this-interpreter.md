### Fixed: the detached lerobot launchers run this interpreter

`lerobot_train(action="start")` and the replay/record/teleoperate/rollout launchers execd a bare `python` from PATH, which does not exist on a host with only `python3`. They now run `sys.executable`; the multi-GPU prefix resolves `accelerate` beside it; a launcher that cannot start is reported as the tool's error.
