### Fixed: refusing an uncalibrated arm no longer leaves its port open, and refuses again

`_connect_robot()` ran lerobot's `connect()` - bus, cameras, `configure()` -
and then refused an arm with no calibration, leaving everything open. The next
call short-circuited on `is_connected` and returned success, so the calibration
gate held for exactly one call, and the process kept the serial port that
`lerobot-calibrate` (the remedy the message names) needed. Measured on an SO-101:
first call refused, second call `(True, "")`. The refusal now closes what the
attempt opened, the calibration check runs on the already-connected path too,
and a bus that an observe action opened is handed back so the driver owns its
whole open sequence.
