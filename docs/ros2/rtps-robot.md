---
description: RtpsRobot and Robot(ros2_transport="rtps") - drive and publish a robot on a ROS 2 graph over pure RTPS, and secure its inbound joint_command surface.
---

# Act as a ROS 2 robot over pure RTPS

[`use_rtps`](../rtps-integration.md) joins a ROS 2 graph as a DDS participant to observe and
publish. The two surfaces here go the other way: they make a robot - simulated by `RtpsRobot`
or physical behind `Robot(ros2_bridge=True, ros2_transport="rtps")` - appear on the graph as
the robot itself, with no `rclpy` and no sourced ROS 2 distro. Both publish through the same
participant, so the samples a real ROS 2 node decodes are hardware's - but the participant
is not a ROS 2 node, and [the graph says so](#what-a-ros-2-node-can-still-tell-apart).

## RtpsRobot: a ROS 2 robot over pure RTPS

`RtpsRobot` is the pure-RTPS sibling of `RosBridgedRobot`. It publishes through
the same participant `use_rtps` does (`strands_robots.rtps.participant`), so it
drives a ROS 2 mobile base with nothing but a pip wheel - and because it
publishes real DDS samples, it can act as the robot itself. A `cmd_vel` command
reaches the shared operator gate whichever of the two asked, under one label, so
an approval or a refusal means the same thing on both.

```python
from strands import Agent
from strands_robots.mesh import RtpsRobot

turtle = RtpsRobot.from_rtps(
    node_name="turtlesim",
    cmd_vel_topic="/turtle1/cmd_vel",
)

turtle.advertise()                       # appear on the graph
turtle.drive(linear=1.0, duration=1.5)   # publish Twist over RTPS for 1.5 s
turtle.stop()

agent = Agent(tools=turtle.tools)        # drive_turtlesim, stop_turtlesim
agent("drive forward for two seconds")
```

See `examples/ros2/rtps_turtle_demo.py` for an end-to-end script, and
`tests_integ/tools/test_use_rtps_live.py` for the gated live test that drives a
real `turtlesim` from a bare participant (`RTPS_LIVE=1 pytest -m rtps`).

For a fully reproducible, self-contained cross-process proof (real turtlesim
node + our publisher, one command), see `examples/ros2/rtps_proof/`:

```bash
cd examples/ros2/rtps_proof
docker compose run --build --rm proof   # exits 0 iff the turtle moved
```

## Hardware bridge over pure RTPS (no rclpy)

`Robot(ros2_bridge=True)` defaults to the rclpy backend (`ros2_transport="rclpy"`,
full `sensor_msgs` fidelity, needs a sourced ROS 2 distro). Pass
`ros2_transport="rtps"` to run the **same bridge over pure cyclonedds** instead -
a single pip wheel, no rclpy and no sourced distro:

```python
from strands_robots import Robot

# rclpy-free: publishes /so101/joint_states (+ camera image_raw) over cyclonedds
# RTPS. Telemetry-only: the inbound /so101/joint_command -> send_action surface
# (ros2_commands=True, the default) drives the arm, so on this transport it needs
# the dds_security_config or explicit opt-out described below to start.
arm = Robot("so101", mode="real", ros2_bridge=True, ros2_transport="rtps", ros2_commands=False)
```

The two transports emit the same topics with byte-identical payloads, so anything that
consumes the **data** - rviz, nav2, a teleop node, `ros2 topic echo` / `ros2 topic pub` -
reads one the same as the other:

```bash
ros2 topic echo /so101/joint_states     # decodes the cyclonedds-published JointState
ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
  '{name: ["shoulder_pan.pos"], position: [0.1]}'   # drives the arm once commands are on (below)
```

The trade-off is the same as `use_rtps`: type coverage is bounded by the IDL
bundle (joint_states + image_raw are in; anything else needs the rclpy backend).
`strands_robots.hardware_rtps_bridge.HardwareRtpsBridge` and `HardwareRosBridge`
derive from one `strands_robots.ros_telemetry.RosTelemetryBase`, which owns the
topic names and the inbound `joint_command` parsing, so both transports present
the identical `publish_joint_states` / `publish_image` / inbound-`joint_command`
surface.

## What a ROS 2 node can still tell apart

The payload is identical; the **graph metadata** is not. A bare DDS participant carries no
ROS 2 node name and no type hash, so a node that inspects the graph rather than the data
sees the RTPS bridge differently. Measured on one domain with a real `rclpy` subscriber
decoding both transports' `/so101/joint_states`:

| read with | rclpy bridge | pure-RTPS bridge |
|---|---|---|
| `ros2 topic echo` | decodes | decodes, field for field identical |
| `ros2 node list` | `/strands_hardware` | absent - not a node |
| `ros2 topic info -v` publisher | `Node name: strands_hardware` | `Node name: _CREATED_BY_BARE_DDS_APP_` |
| `ros2 topic info -v` type hash | `RIHS01_a13ee3a3...` | `INVALID`, plus one `rmw_cyclonedds_cpp` "Failed to parse type hash" warning per subscriber |
| `ros2 topic info -v` history | `KEEP_LAST (10)` (`qos_depth`) | `KEEP_LAST (1)` - the cyclonedds default, no knob |

So a launch file that waits for a node, `ros2 node info`, or tooling that requires a
matching type hash needs the rclpy transport. Everything that subscribes to the topic
works either way.

## Securing the inbound command surface

The inbound `/<robot>/joint_command` subscription lets **any participant on the
DDS domain drive the physical arm**. Two layers harden it, both threaded through
the `Robot()` constructor.

### DDS Security gate (RTPS only)

When the command surface is enabled (`ros2_bridge=True`, `ros2_commands=True`,
`ros2_transport="rtps"`), `HardwareRtpsBridge` **refuses to start** unless one of
the following is true:

- a `dds_security_config` dict is supplied, or
- the operator sets `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` (truthy:
  `1` / `true` / `yes`) to explicitly accept an unsecured graph.

A telemetry-only bridge (`ros2_commands=False`) is publish-only and is **not**
gated. `ros2_commands` is checked rather than read by truthiness: a non-boolean
is refused before any DDS state exists, so a `"false"` from a deployment config
cannot be reported back as "an enabled command bridge" and answered with the
insecure opt-out. `dds_security_config` requires the following keys (each a **non-empty string**:
a path or a `file:` / `data:` URI per the OMG DDS-Security spec); `permissions_ca`
is optional, and held to the same domain when supplied:

```python
from strands_robots import Robot

arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_transport="rtps",
    dds_security_config={
        "identity_ca":  "file:/etc/dds/identity_ca.pem",   # identity CA
        "certificate":  "file:/etc/dds/participant.pem",   # participant cert
        "private_key":  "file:/etc/dds/participant_key.pem",
        "governance":   "file:/etc/dds/governance.p7s",    # signed governance
        "permissions":  "file:/etc/dds/permissions.p7s",   # signed permissions
        # "permissions_ca": "file:/etc/dds/permissions_ca.pem",  # optional
    },
)
```

The credentials are wired into the cyclonedds `DomainParticipant` QoS together
with the builtin DDS-Security plugins, so **both** the outbound telemetry and the
inbound command surface ride an authenticated, access-controlled graph. A
half-filled config is rejected at construction, and so is a well-shaped one whose
credential is not a string: the refusal names the key and what arrived
(`{'private_key': 'NoneType'}`), and no participant is created.

`dds_security_config` is RTPS-specific: passing it with `ros2_transport="rclpy"`
raises, because the rclpy backend gets its DDS Security from the ROS 2 RMW
keystore/env (`ROS_SECURITY_*` / `sros2`), not from a config dict.

### Joint position bounds

`joint_limits={"<motor>.pos": (min, max)}` (threaded into either transport)
range-checks every inbound command. If **any** commanded joint falls outside its
declared range, the **entire** command is rejected - never partially applied - so
one out-of-range joint can never drive part of the arm while the rest holds. Keys
are matched against the joint names the command carries, which are the
`<motor>.pos` names the bridge publishes in `joint_states`, so a controller can
echo them straight back and a key that names no commanded joint constrains
nothing. Joints without a declared bound are unconstrained; to leave a joint unbounded, omit it
rather than declaring an infinite bound. Every bound must be a finite number,
refused at construction.

```python
arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_transport="rtps",
    dds_security_config={...},
    joint_limits={"shoulder_pan.pos": (-3.14, 3.14), "elbow.pos": (-1.57, 1.57)},
)
```

## See also

- [Pure-RTPS ROS 2 integration](../rtps-integration.md) - the `use_rtps` tool, the IDL bundle, topic names
- [Hardware bridge over rclpy](hardware-bridge.md) - the same topics on a sourced ROS 2 distro
- [ROS 2 safety and command gate](safety.md) - what an agent may command on a graph
