### Fixed: `list_robots(mode="real")` names every robot a shipped driver can build

`list_robots(mode="real")` returned 24 of the 49 robots `Robot(name,
mode="real")` can build, hiding every Universal Robots arm, the Franka arms
(`panda`, `fr3`, `fr3_v2`), Spot, Stretch, Kuka iiwa, Kinova Gen3, RBY1,
Unitree H1/H1-2/B2, xArm7 and `open_duck_mini`. Those robots had a native
driver but no `hardware` block in the registry. Each now declares
`hardware.driver: "strands"` (curated entries in `robots.json`, URDF-only entries
stamped by `scripts/build_urdf_registry.py`), so `list_robots(mode="real")`,
`has_hardware()` and `list_driver_coverage()` agree. Driver resolution is
unchanged: these robots already resolved to their native driver.
