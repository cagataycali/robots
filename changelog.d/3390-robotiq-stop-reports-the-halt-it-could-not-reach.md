### Fixed: a Robotiq halt that never reached the gripper is reported

`RobotiqDriver.stop` wrote its own `rGTO`-clearing frame and, when the wire
refused it, called `logger.debug`. The root logger sits at `WARNING` until
something lowers it, so on a default configuration that call emitted nothing and
named no gripper - and `stop` is annotated `-> None`, so no envelope, flag or log
anywhere recorded that the fingers had not stopped. A 2F-85 whose halt does not
land keeps travelling to the last commanded aperture, closing on whatever is
between the fingers or opening and releasing it.

`stop` now delegates to `stop_task`, which already decided a verdict for the
identical write, and logs a non-success through `halt_failure_detail` at `error`
naming the gripper and what the wire said - the shape `BoosterDriver`,
`EarthRoverDriver` and `CrazyflieDriver` already use, and the one
`HardwareDriver.stop` documents. Behaviour on a disconnected gripper is
unchanged: `stop_task` answers `success` there, so nothing is logged.

`tests/drivers/test_stop_hook_logs_the_halt_it_could_not_complete.py` graded only
the hooks that delegate to an envelope verb, which is why a hook holding its own
wire was invisible to it. It now carries a second derived relation: a `stop` that
catches its own wire failure must report it at a level a default configuration
emits, naming the robot.
