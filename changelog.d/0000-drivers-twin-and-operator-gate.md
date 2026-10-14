### Added: a `Robot` on a bus, its twin, and the operator gate

`Robot("so101", port=...)` drives an SO-100/101 Feetech bus and reads and writes LeRobot's
calibration JSON unchanged; `Robot("g1"|"go2"|"h1", port=<iface>)` reads Unitree DDS and
`Robot("microduck", port=<robotd socket>)` reads the duck's robotd, both observe-only. A hardware
`Robot` adds a twin to `Simulation.default()` that mirrors every `act` and every measured
`observe`. `Robot(confirm=callable)` is the operator gate: `confirm_act` on a real body writes
nothing unless the callable says yes.
