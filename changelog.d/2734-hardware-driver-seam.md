### Added: choose which driver builds a real robot

`Robot(name, mode="real")` built exactly one thing - `strands_robots.hardware_robot.Robot`,
which constructs a lerobot `RobotConfig` and wraps a lerobot driver. That is the right driver
for every robot in the shipped registry and the wrong shape for a robot lerobot does not
model: a humanoid with its own state machine, a rover reporting GPS, a base publishing a
point cloud. There was no seam to reach anything else through, and nothing recorded which
members of that 3000-line class the mesh, the teleop rail and the agent tool surface actually
rely on - so a driver author had to read it to find out.

`driver=` now selects the implementation: `"auto"` (the default) honours a robot's registry
`hardware.driver` and otherwise builds the lerobot driver, `"lerobot"` pins that path, and
`"strands"` builds the native driver registered for the robot via
`strands_robots.drivers.register_native_driver`. Asking for a native driver that is not there
is refused by name, listing the robots that do have one and both remedies, rather than being
quietly served the lerobot driver - a caller who got the substitute would debug the wrong
robot. The value is checked in every mode, so a typo is not accepted by whichever branch
happens not to read it, and a driver name added to the vocabulary without a route in the
factory is refused instead of taking a neighbour's branch.

`strands_robots.drivers.HardwareDriver` writes the contract down, and it is the measured one
rather than an aspirational one: `get_observation` is not a member (the mesh reads it from the
driver's inner device), and neither are the sensor attributes a mesh publishes (`_pose`,
`_imu` and their siblings are all read with a `getattr` default, so a driver with no lidar is
complete). `register_native_driver` refuses a class that does not satisfy it and names the
members it lacks, so a driver missing `stream` fails where it is registered rather than on
the first agent call. Both drivers take the same post-build path onto the mesh, because a
robot that is built but never published is one no peer can see.

The default path is unchanged, and that is what the tests pin hardest: not mentioning
`driver`, `driver="auto"` and `driver="lerobot"` build the same class with the same lerobot
config.
