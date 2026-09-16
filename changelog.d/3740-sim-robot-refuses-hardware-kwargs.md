### Fixed: a hardware-only keyword on a simulated `Robot()` is refused, naming the mode

The sim backend drops any keyword it does not own, by design, so one call can
carry another backend's options (`num_envs`, `device`). That tolerance also
swallowed `port=` - the one keyword that only ever means "a physical robot":
`Robot("so101", port="/dev/cu.usbmodem…")` with `mode="real"` forgotten built
a simulator under `status=success` while the arm on the desk stayed still.
The real branch already reports the sim-only spawn keywords it cannot honour
(`position`, `orientation`, `keyframe`); this is the mirror.

`Robot(mode="sim")` now refuses every name the hardware class forwards to a
lerobot config (`port`, `robot_ip`, `kp`, `kd`, `calibration_dir`, ...) with a
`TypeError` that lists the keywords supplied and the remedy - add
`mode='real'`, or drop them - and says how the call became a simulation (the
default, or `mode='auto'` finding no servo bus). What the set has in common is
the driver, not the physicality, and the message says so: `mock=` asks for a
mocked servo bus and `is_simulation=` points the lerobot driver at a simulator,
so neither describes a physical robot, but neither has a driver to configure
under `mode="sim"` either. Cross-backend options pass through untouched.
