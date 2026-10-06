### Fixed: the dashboard adds a networked robot (G1, Go2, UR, Spot...) by its address

The Devices sheet treated every real robot as a servo bus. `unitree_g1` was listed
as real-capable, but an IP was refused as "must be a path under /dev/", and the
form offered only a serial picker and a lerobot calibration id. The new
`strands_robots.drivers.port_kind(name)` says what `port=` names for the driver
that builds a robot (`"serial"` or `"address"`). The spawn refusals and the form
branch on it: a networked robot takes an optional address, plus a network
interface when its driver binds one, skips the bus-claim probe, and shows the
one-line `Robot(..., mesh=True)` to run on the robot's own computer instead.
`/api/robots/registry` reports this as `real_transport`, also for robots built
by an undeclared native driver.
