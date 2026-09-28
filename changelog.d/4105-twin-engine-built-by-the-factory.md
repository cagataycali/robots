### Changed: the twin transport's engine is built by `Robot()`, not by the driver

`Robot(..., mode="real", transport="twin")` now builds the MuJoCo engine and
hands it to the driver as `sim=`; `FeetechTwinBus` and `M3ProTwinGraph` take
`sim` as a required argument and never destroy it, so the engine outlives
`cleanup()` and a second `connect_eagerly()` binds the same one. A bare
`FeetechDriver(transport="twin")` or `YahboomM3ProDriver(transport="twin")`
without `sim=` is refused naming the `Robot(...)` call that builds it. A build
failure is raised by `Robot()` instead of reported by the first connect. No
driver imports `strands_robots.simulation` any longer: two of the three
deferred upward imports in the layer graph are gone (#3818).
