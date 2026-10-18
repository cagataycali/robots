### Fixed: a NaN never reaches the bus; a mislabelled driver, a LeRobot 0.6 calibration and a second render thread

A NaN target used to pass `coerce()` and, on a Feetech arm, round to `range_min`: a full-speed move to the end
stop. Every non-finite target now raises `NonFiniteTarget` (a `ValueError`) before any write, and a `run()`
whose policy emits one ends with `stopped_by="fault"`. A `Driver` instance whose `names` are not the
description's `action_keys`, in order, is refused, and the `Driver` protocol now names
`supports_position_hold`, `calibration` and `resolution`. `Robot(..., calibration=path)` takes a LeRobot file,
the default search finds LeRobot 0.6's `so_follower/<id>.json`, and a missing file lists where it looked.
`calibrate(interactive=True)` with no terminal refuses before torque goes off. Every MuJoCo render runs on one
GL thread, so a background `start()` after a `run()` no longer fails (EGL) or hangs (macOS).
