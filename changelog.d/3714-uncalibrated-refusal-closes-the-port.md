### Fixed: refusing an uncalibrated arm no longer leaves its port open, and refuses again

`_connect_robot()` ran lerobot's `connect()` - bus, cameras, `configure()` -
and then refused an arm with no calibration, leaving everything open. The next
call short-circuited on `is_connected` and returned success, so the calibration
gate held for exactly one call, and the process kept the serial port that
`lerobot-calibrate` (the remedy the message names) needed. Measured on an SO-101:
first call refused, second call `(True, "")`. The refusal now closes what the
attempt opened, the calibration check runs on the already-connected path too,
and a bus that an observe action opened is handed back so the driver owns its
whole open sequence. Hand-back, `is_connected` and `connect()` are one unit
under the device lock, on both the policy path and the teleop loop's lazy
connect, and the observe actions check-and-open under the same lock: an observe
action is ungated by design, and one that landed in the thread hop between the
hand-back and the `is_connected` read reopened the bus, so on a camera-less arm
the read answered True and `connect()` was skipped - `(True, "")` with
`configure()` never run. It now waits for the bring-up and finds the arm up.

The gate also asks its question the way `status` now does - on the class. On the
instance, `hasattr(robot, "is_calibrated")` *evaluates* the property, so a
lerobot arm was swept for homing offsets and ranges twice per connect; and
`hasattr` swallows an `AttributeError` raised *inside* that read, which is
indistinguishable from "this driver has no such property". A driver whose
`is_calibrated` is `self.bus.is_calibrated` over a bus built lazily raises
exactly that, and the gate was then skipped altogether: `(True, "")`, cameras
open and `configure()` already run, for an arm whose calibration was never
checked. The check now always runs, and a property that cannot answer refuses
the arm instead of silently permitting motion.
