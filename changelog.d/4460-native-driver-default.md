### Changed: the native strands drivers are the default for real hardware

`Robot(name, mode="real")` with no `driver=` now builds the native driver
registered for the robot (`FeetechDriver` for `so101`/`so100`/`lekiwi`/`koch`,
`URDriver` for `ur5e`, `FrankaDriver` for `panda`, ...) and falls back to
lerobot only for a robot this package has no driver for (`omx`, `openarm`,
`reachy2`). A registry `hardware.driver` declaration still outranks the default
(`earthrover` declares `"lerobot"` because its documented teleop reads live on
the lerobot wrapper) and `driver="lerobot"` still pins that path. An SO-101 on a
fresh install therefore connects without the `[lerobot]` extra or a lerobot
calibration file, and an arm lerobot already calibrated keeps that file: the
native driver loads `<lerobot calibration dir>/robots/<type>/<tool name>.json`
when `calibration=` is not given, the same file the lerobot path read with no
`id=`; a dashboard spawn with a remembered `robot_id` goes through the same
lookup, so a missing file or a missing lerobot keeps the uncalibrated bring-up.
`strands_robots.registry.NATIVE_DRIVER` names the new
default; `DEFAULT_DRIVER` keeps naming the fallback.
