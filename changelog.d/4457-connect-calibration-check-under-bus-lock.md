### Fixed: a lerobot arm's connect no longer fails with "Port is in use!" on a mesh peer

`HardwareRobot._connect_robot` ran `connect()` under the device's bus lock but
read `is_calibrated` -- on a Feetech arm a serial read of every servo's position
limits -- with no lock. By then `is_connected` was True, which is what the mesh's
joint probe and camera publisher wait for, so the two readers met on the SDK's
single port handler, the connect rolled back and the log blamed a bus that "did
not open". The calibration check now runs under `bus_lock(robot)` like every
other reader.
