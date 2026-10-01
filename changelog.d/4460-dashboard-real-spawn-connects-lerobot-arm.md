### Fixed: a real arm spawned from the dashboard now actually connects

The Devices sheet's `mode=real` spawner (and the Deploy tab's generated
snippet) brought the arm up through `robot.connect_eagerly()`, a method only
the native drivers (G1, UR, Robotiq, Reachy) have. The lerobot-backed
`HardwareRobot` that `Robot("so101", mode="real")` builds never had it, so
every SO-10x spawn printed `eager connect failed ... 'Robot' object has no
attribute 'connect_eagerly'`, stayed `connected: false` on the fleet and
published no joints or camera frames until a task was run on it. Both scripts
now use `connect_eagerly()` when the driver has it and `_connect_robot()`
otherwise, so an uncalibrated arm is refused with the real reason (and the
port is released for `lerobot-calibrate`) and a calibrated one streams at once.
