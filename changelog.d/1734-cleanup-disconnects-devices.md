### Fixed: `cleanup()` now closes the motors bus and the cameras

`Robot.cleanup()` released every resource the library owns -- the shutdown latch,
a teleop loop, a `RUNNING` task, the task executor, the mesh client, the ROS
bridge -- but never called `robot.disconnect()`, so the only resources that are a
physical device were the only ones it left open: the motors bus serial port plus
a `/dev/video*` node and a read thread per camera. A serial port is exclusive, so
a second process -- or a re-constructed `Robot` in the same one -- could not open
the same `/dev/tty*`, which made the documented recovery for a wedged arm (tear
down, reconnect) unavailable without exiting the process. `cleanup()` is also
what `__del__` calls, so a script that simply ended left the arm energised at its
last commanded position instead of going through the driver's disconnect, where
torque disable and gripper release live; and it was unrecoverable afterwards,
because the executor is down and the shutdown is latched, while lerobot's gated
`Robot.disconnect()` refuses in every half-open state. `cleanup()` now
disconnects, preferring the driver's own `disconnect()` while the robot is
connected and otherwise closing each device individually -- so a half-open set,
or one abandoned partway through lerobot's single unguarded disconnect loop,
still ends with the port released and every camera closed, and a close that
raises is warned rather than propagated. The disconnect runs last, after the
teleop loop, the executor drain, the mesh and the ROS bridge, because
`send_action` re-opens the robot lazily on a command that finds it disconnected:
a port closed while any command source is still live would be re-opened behind
the teardown and then stay open for the life of the process.
