### Bug Fixes

- **tools**: `lerobot_calibrate` refuses an `overwrite` that is not a boolean instead of reading
  it by truthiness. On the `restore` action the flag selects a posture - keep a calibration
  already at the destination, or replace it with the backup's copy - and every non-empty string
  is truthy, so a caller who spelled the opt-out selected replace. Measured on `f7950da5` with
  the destination holding `homing_offset=1` and the backup holding `999`: `overwrite="false"`,
  `"no"` and `"0"` each returned `status="success"`, wrote `999` over `1`, and reported
  `Overwrite mode: false` beside the file just replaced. Restoring is the path a lost
  measurement is recovered on, so the file replaced is usually one the operator cannot
  re-measure. Both surfaces that read the flag now consult the shared `boolean_flag_error`
  domain: the tool for the `restore` action only, so an action that ignores the flag is never
  refused for it, and `LeRobotCalibrationManager.restore_calibrations` at the read, ahead of
  resolving the backup directory, so a refused flag leaves nothing touched. `True` and `False`
  select exactly what they did before, numpy booleans included. Towards #3356.
