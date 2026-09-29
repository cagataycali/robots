### Changed: the ROS 2 mobile bases move from `strands_robots.mesh` to `strands_robots.drivers.ros`

`MobileBaseRobot`, `RosBridgedRobot`, `RosbridgeRobot`, `RtpsRobot` and
`AckermannRosRobot` drive a robot; they do not network one, so they now live
with the other drivers. Import them from `strands_robots.drivers.ros`. The old
names - `from strands_robots.mesh import RtpsRobot` and the module paths
`strands_robots.mesh._mobile_base`, `.ackermann_robot`, `.ros_bridge`,
`.rosbridge_robot`, `.rtps_robot` - still resolve to the same objects with a
`DeprecationWarning` and are removed in 0.7. The three private copies of the
refusal envelope in those modules now use `strands_robots.drivers.base.refuse`.
