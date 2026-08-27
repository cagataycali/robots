### Fixed: the mesh reads joints from a driver that owns its bus

A robot reaches its motors one of two ways. A lerobot robot is a wrapper: `hardware_robot.Robot`
holds the device that owns the bus under `robot`, and the wrapper answers no read itself. A native
driver owns the bus directly, so it *is* the device.

`Mesh._read_state` resolved only the first shape, so a driver satisfying every member of
`DRIVER_SURFACE` published no joint telemetry at all. Every other section reaches the mesh through a
`getattr(robot, name, None)` read straight off the driver -- `_imu`, `_battery`, `_temps` and
thirteen siblings -- so such a driver could report how hot a joint was and not where it was, while
`missing_driver_members` reported no problem and the presence heartbeat kept advertising the peer.
That is the presentation `_read_state`'s own docstring exists to avoid: a peer that "went silent on
the state topic while its presence heartbeat kept advertising it -- indistinguishable from a peer
whose state thread had died".

`read_joints` could already read such a driver unchanged, preferring `bus.sync_read` and falling
back to `get_observation`; nothing handed the driver to it. So the repair is to resolve the
telemetry device rather than assume it. `bus_access.joint_read_source` prefers `robot.robot` and
falls back to the robot itself, admitting a device exactly when `read_joints` has a route to it --
the admission rule derived from that function's own branch rather than restated, so a caller cannot
admit a device the reader raises on, or refuse one it could have read.

Deriving the rule also repairs a narrower disagreement at the same site. The old gate required
`get_observation`, which is the capability `read_joints` treats as its *fallback*, and refused a
device carrying only the `bus` it *prefers* -- the shape an SO-100/SO-101 driver has, where joint
position, velocity and current are the whole telemetry.

An inner device is preferred whenever one is present, so a wrapper is never read in place of the
device it wraps and a robot that already published joints resolves to the same device as before.
`is_connected` remains the liveness gate, and a driver with no motors to report still publishes no
`joints` without appearing as a failed probe in `degraded`.

Scoped to the state topic's `joints` section. Two further consequences of a driver having no inner
device are left for their own change: presence drops `connected` and `hw`, and
`_publish_cameras_once` routes a real peer into `_publish_sim_cameras()`.
