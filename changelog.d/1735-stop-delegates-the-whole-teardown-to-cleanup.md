### Fixed: `stop()` is terminal for a robot that is not fully connected

`hardware_robot.Robot.stop()` disconnected the driver itself before delegating
to `cleanup()`, and did so unguarded. lerobot gates `Robot.disconnect()` on
`is_connected` (`bus.is_connected and all(cam.is_connected ...)`) and raises
`DeviceNotConnectedError` when it is false, so that call raised for any robot
that was not *fully* connected - one that never connected, or one a failed
bring-up left disconnected. `stop()` is fail-soft for an operator, so the raise
was logged and `cleanup()` was never reached: no shutdown latch, a task
executor still accepting work, and any device a failed close had left open still
held, with no entry point remaining that would close it. A rollout still in
bring-up was then not truncated either, and reported itself `completed`.

`stop()` now performs no teardown step of its own and delegates the whole of it
to `cleanup()`, off the event loop - `cleanup()` joins the task executor and
closes a serial port, both of which block, and it was being awaited inline. That
also restores the documented ordering: the devices close only after the teleop
loop, the task executor, the mesh and the ROS bridge are down, because
`send_action` re-opens the robot lazily on a command that finds it disconnected.
