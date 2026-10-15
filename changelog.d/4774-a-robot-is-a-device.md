### Added: a Robot is a device

Every `Robot` now publishes a manifest generated from its `Description` and the registry: `device_id`,
`device_type`, labels, sensors and controls (each with its range, its safe band and what happens on disconnect),
rpcs and events. Every write goes through one safety chain (range, safe band, interlocks, validators, then the
operator gate for a target outside the safe band); `estop()` bypasses the chain and latches until `resume()`, and a
watchdog holds or releases a body whose controller goes quiet. `Driver.hold()` and `supports_position_hold` are
declared per driver, and a LeRobot remote policy refuses a body that cannot hold position. Feetech joints read in
exact degrees through the bus calibration, matching LeRobot's own conversion.
