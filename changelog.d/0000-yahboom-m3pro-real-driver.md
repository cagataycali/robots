### Added: `yahboom_m3pro` drives the real Yahboom ROSMASTER M3 Pro over its ROS 2 graph

`Robot("yahboom_m3pro", mode="real")` now builds `YahboomM3ProDriver`
(`strands_robots/drivers/yahboom_m3pro.py`), the registry entry declaring
`hardware.driver = "strands"` so no `driver=` keyword is needed and
`list_driver_coverage()["yahboom_m3pro"]` reads `("strands",)`. The robot's
motors sit behind a micro-ROS expansion board whose topics the vendor's own
stack speaks, so the driver speaks them too: `/cmd_vel` (`geometry_msgs/Twist`,
`linear.y` is the mecanum strafe) for the base, `/arm6_joints`
(`arm_msgs/ArmJoints`: six integer-degree servos + `time` ms) for the arm,
`/odom_raw` and `/imu/data_raw` for the reads. Two transports, `rosbridge`
(default; `port=` is the bridge's `host[:port]`, reachable from any host with
the `[rosbridge]` extra) and `ros2` (in-process `rclpy`, for a driver on the
robot), both forwarded through the package's existing `use_rosbridge` /
`use_ros` transports - so every `/cmd_vel` write passes the shared operator
gate and `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel` is the headless pre-approval.

`send_action` speaks the twin's vocabulary - `arm1.pos .. arm5.pos` and
`gripper.pos` in radians, `linear.x` / `linear.y` / `angular.z` in SI - and
converts at the wire with the inverse of the keyframe the MJCF was written
with (`deg = 90 + degrees(q)`), which lands the URDF ranges on the servo
ranges exactly (0-180 for servos 1-4, 0-270 for servo 5, the gripper crank's
`-1.54..0` on 30..180); `joint_signs` flips a servo the bench shows reversed.
Out-of-range targets and twists past the vendor teleop ceiling are refused by
name, never clamped. `move()` requires `duration_s`, streams the twist at
10 Hz above the firmware's ~0.3 s watchdog and ends in an explicit zero,
reporting both halves; `cleanup()` is a stop first. `connect_eagerly()`
proves `/cmd_vel` and `/arm6_joints` are on the graph and, when only
`/parameter_events` and `/rosout` are, names the micro-ROS agent handshake as
the cause and its restart as the remedy. `get_observation()` is `{}`: the
board publishes no arm joint-state topic the driver has verified, and a
command is not a reading. Agent verbs: `status`, `sensors`, `arm`, `gripper`,
`home`, `move`, `stop`.

Tests: `tests/drivers/test_yahboom_m3pro_driver.py` - network-free against a
recording transport double, covering the unit maps both ways, every refusal
before the wire, the gate (a base command with no approval never reaches the
recorder), the held move's trailing stop, the connect diagnostics and every
agent verb; the registry test flips from "no hardware block" to "the hardware
block names the native driver". `docs/robots/mobile.md` documents the
interface, the transports and what the driver deliberately leaves out.
