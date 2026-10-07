### Fixed: `list_robots(mode="real")` includes native-driver robots

`list_robots(mode="real")` returned 24 of the 49 robots a user can drive for
real, hiding every robot whose hardware fact lives in `_NATIVE_DRIVERS` rather
than the entry's `hardware` block — every Universal Robots arm (ur3e/5e/7e/
8long/10e/12e/15/16e/18/20/30), Franka Panda (`panda`/`fr3`/`fr3_v2`), Spot,
Stretch (1 + 3), Kuka iiwa, Kinova Gen3, RBY1, Unitree H1 (1 + 2), xArm7, b2
and open_duck_mini. `Robot(name, mode="real")` built a working driver on each
of them, so the listing was silently wrong against the factory. The filter now
honours native-driver registration too, matching the row set
`list_driver_coverage` has surfaced all along; the docstring's claim "joins
the two halves of `list_driver_coverage`" is now the implementation. The
single-point reader `has_hardware` keeps its declaration-only contract (its
docstring already names the split).
