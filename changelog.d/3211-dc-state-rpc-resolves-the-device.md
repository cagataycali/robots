`RobotDeviceDriver.getState` reported no `joints` key at all for a robot whose
motors it could already read. A robot reaches its motors one of two ways -- a
lerobot robot is a wrapper holding the device that owns the bus under `robot`, a
native driver owns its bus directly and so *is* the device -- and this RPC
resolved only the first shape, `getattr(self._robot, "robot", None)` behind a
`hasattr(inner, "get_observation")` gate. That is the same resolution that left a
native driver publishing no joint telemetry on the mesh state topic (#2749); when
the read here was converted to `bus_access.read_joints` (#2666) the resolution
beside it was left as it was, so a native driver answered under the same
successful status a readable arm gets. The gate also demanded the capability
`read_joints` treats as its *fallback* while refusing a device carrying only the
`bus.sync_read` it prefers.

15 of the 25 registered native drivers carry a joint-read route and none of them
could answer this RPC. The five `FeetechDriver` robots (`so100`, `so101`,
`lekiwi`, `hope_jr`, `open_duck_mini`) hit both halves.

The device is now resolved by `bus_access.joint_read_source`, the one owner of
that question. An inner device is still preferred whenever one is present, the
read still takes the shared motor-bus lock, and a robot that can answer no joint
read is still reported without a `joints` key rather than as an error.
