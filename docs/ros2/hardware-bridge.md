---
description: Robot(ros2_bridge=True) - publish a real robot's live observation on a ROS 2 domain and, with ros2_commands, accept joint_command over it.
---

# Hardware bridge: publish a real robot on a ROS 2 domain


The hardware `Robot` is the symmetric counterpart of the sim bridge: construct
it with `ros2_bridge=True` and it owns a
`strands_robots.hardware_ros_bridge.HardwareRosBridge` that advertises the real
arm's live observation on a ROS 2 domain. The sim bridge
(`SimRosBridge`) and the hardware bridge (`HardwareRosBridge`) are thin
subclasses of the same `RosTelemetryBridge`, and the pure-RTPS transport
(`HardwareRtpsBridge`) shares the same wire contract through their common
`RosTelemetryBase`, so a physical arm and its digital twin publish **identical
topics** - a simulated robot and the real one it mirrors are indistinguishable on
the ROS 2 graph:

| Topic | Direction | Type | Content |
|-------|-----------|------|---------|
| `/<robot>/joint_states` | published | `sensor_msgs/msg/JointState` | joint names + positions, every control step |
| `/<robot>/<camera>/image_raw` | published | `sensor_msgs/msg/Image` (`rgb8`) | one frame per camera |
| `/<robot>/joint_command` | **subscribed** | `sensor_msgs/msg/JointState` | inbound `name`/`position` -> `send_action`, drives the real arm |

`name` and `position` are one table read by index. A robot whose observation
does not carry every joint of `robot_joint_names()` - every floating-base
humanoid, quadruped and mobile base, whose root freejoint is joint 0 and is not
an observation key - publishes only the joints it observed, so the two arrays
stay the same joints. A caller that hands `publish_joint_states` a differing
number of names and positions has no pose to publish: the message is dropped
whole with a warning naming both counts, rather than published with every joint
after the gap under its neighbour's name.

The first two are **outbound telemetry** (shared, byte-identical, with the sim
bridge). The third is the **inbound command** surface that makes the hardware
bridge *full duplex*: an external ROS 2 node (a teleop joystick node, MoveIt, a
trajectory replayer, or the agent's own `use_ros(action="publish", ...)`) can
publish a `JointState` to `/<robot>/joint_command` and the bridge forwards each
message straight into `Robot.send_action({motor.pos: float})`. Because the
command topic carries the *same* joint names the bridge publishes in
`joint_states`, a controller can echo our names straight back to drive the arm.
The sim sibling does not subscribe - a simulation is driven by its physics
engine; only the real arm is the thing on the graph an external controller can
physically move.

```python
from strands_robots import Robot

# Opt in to the bridge; the arm's observation is mirrored on ROS 2 domain 0.
arm = Robot("so101", mode="real", ros2_bridge=True, ros2_domain=0)

# Each control step of a running task publishes joint_states (+ camera frames).
# Or publish the current observation on demand without starting a task:
arm.publish_ros_observation()                 # joints + cameras
arm.publish_ros_observation(skip_images=True)  # joints only (opt out of cameras)

# Full duplex: with the default ros2_commands=True the bridge also subscribes to
# /so101/joint_command and forwards each message into Robot.send_action, so an
# external ROS 2 node can drive the real arm:
#
#   ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
#     '{name: ["shoulder_pan.pos", "elbow.pos"], position: [0.1, -0.2]}'
#
# For a read-only telemetry bridge (no inbound control), opt out:
arm_ro = Robot("so101", mode="real", ros2_bridge=True, ros2_commands=False)

# rclpy-free: run the SAME bridge over pure cyclonedds (no sourced ROS 2
# distro). Byte-identical topics; type coverage bounded by the IDL bundle.
# Telemetry-only: on this transport the inbound command surface refuses to
# start without a dds_security_config or the explicit opt-out (see below).
arm_rtps = Robot("so101", mode="real", ros2_bridge=True, ros2_transport="rtps", ros2_commands=False)
```

External ROS 2 nodes - rviz, nav2, or the agent's own `use_ros` calls - then see
the physical robot as a live participant:

```bash
ros2 topic list | grep so101          # /so101/joint_states, /so101/<cam>/image_raw
ros2 topic echo /so101/joint_states   # live joint positions from the real arm
```

The bridge is **opt-in**: `ros2_bridge=False` (the default) never touches ROS 2,
so a robot only becomes a ROS 2 device when an operator explicitly asks for it -
the same safety stance as `Robot(mode="sim")` being the default. When `rclpy` is
missing, `ros2_bridge=True` raises an `ImportError` at construction naming both
routes forward: sourcing a ROS 2 distro, or `ros2_transport="rtps"`, which
publishes the same topics over the pip-installable cyclonedds binding and needs
no distro at all. The
inbound command path is on by default (`ros2_commands=True`); set
`ros2_commands=False` for a read-only telemetry bridge that publishes but cannot
be driven. Only a boolean names either posture - `ros2_bridge` and
`ros2_commands` are checked at construction, so a config that spells the flag
`"false"` is refused rather than reading as the truthy value it is and opening
the surface it asks to close. A daemon thread spins the node so inbound commands are serviced
concurrently with publishing, and it is torn down cleanly on `cleanup()`/`stop()`.
That teardown is best-effort: a node destroyed on a context another
component already shut down is reported at WARNING and `cleanup()` carries
on to disconnect the motors bus and the cameras, because a bridge that will
not release must not leave the serial port held or the arm energised. The same
rule holds one level in, where the bridge releases two things - its node handle
and, when it was this bridge that called `rclpy.init()`, the process-wide
context: a failure releasing one no longer skips the other, so a node that
refuses to be destroyed does not leave the participant on the domain for the
life of the process. A context that itself refuses to shut down is logged at
warning, because nothing after it retries.

Because the inbound `joint_command` topic drives the physical arm, two guards
harden it (both threaded through `Robot()`):

- `joint_limits={"<motor>.pos": (min, max)}` range-checks every inbound command;
  if any commanded joint is outside its declared range the **entire** command is
  rejected (no partial application). Keys are matched against the joint names
  the command carries - the same `<motor>.pos` names the bridge publishes in
  `joint_states` - so a key that names no commanded joint constrains nothing,
  and joints without a declared bound are unconstrained. Every bound must be a finite number - a non-finite one declares
  a range that admits nothing, so it is refused at construction. Available on
  both transports.
- For the pure-RTPS transport (`ros2_transport="rtps"`), a `dds_security_config`
  (or the explicit `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` opt-out) is
  **required** to expose the command surface - see the
  [RTPS integration guide](../rtps-integration.md#securing-the-inbound-command-surface).
  rclpy DDS Security is configured at the RMW layer (`ROS_SECURITY_*` / `sros2`),
  not by a config dict.

See `examples/ros2/hardware_bridge_demo.py` for a runnable end-to-end script.

![Hardware ROS 2 bridge: an SO-101 camera frame published by HardwareRosBridge and received by an independent ros2 subscriber over DDS, byte-identical](../assets/hardware_ros_bridge_proof.png)

The frame above was rendered for an SO-101, published on `/so101/wrist/image_raw` by `HardwareRosBridge` over real DDS, and decoded back by a separate `rclpy` subscriber - byte-identical round trip. On the same run `ros2 topic echo /so101/joint_states` returns the live joint vector, so the robot is a first-class ROS 2 device on the graph.

