### Added: `Robot`, a body in a world

`Robot("so101")` joins `Simulation.default()` (or `sim=`) with `observe`, `act`, `run`,
`start`/`stop`, `set_pose`, `set_gains` and `describe`. `run()` drives it with a `Policy`, anything
with `act`/`reset`, or a bare callable, checks the policy's requirements before step one, and returns
a `RunResult`; `Simulation.run` steps many robots in lock-step.
