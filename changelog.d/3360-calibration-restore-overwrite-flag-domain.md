### Bug Fixes

- **tools**: a calibration restore refuses an `overwrite` posture that is not a boolean
  instead of reading it by truthiness. `overwrite` is a confirmation gate in front of the
  one write in `lerobot_calibrate` that destroys a measurement, and it was read as
  `if dest_file.exists() and not overwrite: continue` - so `not "false"` was `False` and
  every string spelling of the opt-out (`"false"`, `"no"`, `"off"`, `"0"`) overwrote each
  existing calibration, while the tool answered `status="success"` with the text
  `Overwrite mode: `false`` beside a restored count of 1. A calibration is a physical
  measurement of one arm's homing offset and travel limits; nothing in the backup can
  reconstruct the one it replaced, and the atomic commit that protects a *failed* write
  does not help here because the overwrite succeeded. `None`, `0` and `""` took the skip
  branch without being a declared spelling of it. Both surfaces that read the flag now
  check it against the shared `boolean_flag_error` domain - the documented
  `LeRobotCalibrationManager.restore_calibrations`, ahead of the backup read so no refusal
  arrives after the file it was protecting is gone, and the `lerobot_calibrate` facade,
  so the refusal names the parameter rather than surfacing as a tool crash. The two
  honoured postures are unchanged.
