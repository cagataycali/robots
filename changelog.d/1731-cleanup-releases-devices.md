### Fixed: `cleanup()` disconnects the robot's devices instead of holding them for the life of the process

`Robot.cleanup()` released every resource the library owns - the shutdown latch,
the task executor, the mesh client, the ROS bridge - but never the ones that are
physical devices: the motors bus serial port and one `/dev/video*` node plus read
thread per camera. Both API tables already documented it as closing the cameras.
A serial port is exclusive, so the port stayed held for the life of the process
and the documented tear-down-and-reconnect recovery could not work without
exiting; `cleanup()` is also what `__del__` calls, so a script that finished
normally left the arm energised at its last commanded position rather than going
through the driver's own disconnect, where the Feetech
`disable_torque_on_disconnect` write lives. Nothing could recover it afterwards:
the executor is shut down and the shutdown latched, so no library entry point
remained that would reach a disconnect.

`cleanup()` now disconnects the devices, after the task executor has drained so a
rollout still finishing cannot command a port closing underneath it, preferring
the driver's own `disconnect()` and then closing each device independently - so a
half-open set, which that driver call refuses to touch because it is gated on
every device, is still released, and one camera that will not close cannot keep
the bus open.

`stop()`, the async spelling of the same teardown, called
`robot.disconnect()` *before* `cleanup()`. That call raises
`DeviceNotConnectedError` when the robot is not connected, so stopping a robot
that was never connected ended the teardown early and left the executor running
and the shutdown unlatched - the terminal guarantee did not hold for the most
ordinary case. It now delegates to `cleanup()`, which owns the disconnect.
