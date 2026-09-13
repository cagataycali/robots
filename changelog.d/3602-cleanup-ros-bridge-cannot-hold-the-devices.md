### Fixed: a ROS 2 bridge that will not shut down no longer keeps the devices open

`Robot.cleanup()` guards every best-effort teardown step where it is called -
the teleop loop, the running task, the mesh client - so that a software
resource which will not release cannot decide whether the physical ones do. The
ROS 2 bridge shutdown was the one step that was not guarded, and it is the last
step before `_disconnect_devices()`. A `destroy_node()` on a context another
component had already shut down (`HardwareRosBridge.shutdown()` clears its
handle in a `finally` but adds no `except`) therefore escaped to the handler at
the bottom of `cleanup()` and skipped the disconnect altogether.

Measured with a driver double on a one-camera arm, bridge shutdown raising
`invalid handle`: the serial port stayed held, the bus and the camera stayed
open, the driver's own `disconnect()` - where torque disable and gripper
release live - was never called, and the only report was
`ERROR: Cleanup error`, which names the bridge failure but not the leak. A
serial port is exclusive, so the port stayed held for the life of the process
and the arm stayed energised at its last commanded position, with no library
entry point left that would close either: the executor is already shut down and
`_shutdown_event` is set. The leak recovered only for a robot nobody holds, and
only once `__del__` ran a second `cleanup()` that found the handle already
cleared.

The call is now guarded like its siblings: the failure is reported at WARNING
naming the step, the bridge handle is still cleared, and the devices close. The
sim engine already wrapped this same call in `contextlib.suppress`, so the guard
was missing on the one path where the consequence is physical.
