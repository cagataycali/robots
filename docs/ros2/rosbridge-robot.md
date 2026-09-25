---
description: RosbridgeRobot - drive a ROS 1 or remote mobile base over a rosbridge WebSocket under the fleet-standard drive contract, plus the NASA Curiosity quickstart.
---

# Drive a robot over rosbridge

## RosbridgeRobot

For mobile bases that expose the standard `cmd_vel` / odometry / scan topic
trio, `RosbridgeRobot` wraps that wiring so a remote ROS 1 or remote robot
drives like any other strands robot - the same `Agent(tools=[robot])` pattern
used for simulated and hardware arms. It forwards through the same transport
`use_rosbridge` does (`strands_robots.rosbridge`), so a `cmd_vel` command reaches
the shared operator gate whichever of the two asked, under one label: an approval
or a refusal means the same thing on both.

### Constructor

```python
from strands_robots.mesh import RosbridgeRobot

robot = RosbridgeRobot(
    node_name="my_robot",
    cmd_vel_topic="/cmd_vel",
    odom_topic="/odom",
    scan_topic="/scan",  # optional
    host="192.168.1.20",
    port=9090,
    cmd_vel_type="geometry_msgs/Twist",  # defaults; matches ROS 1
    odom_type=None,  # auto-resolved via rosapi when omitted
    scan_type=None,
    max_linear=2.0,  # m/s clamp
    max_angular=1.0,  # rad/s clamp
    max_duration=30.0,  # longest accepted drive() hold
    publish_rate=10.0,  # Hz
)
```

All parameters are optional except `node_name`, `cmd_vel_topic`, and `odom_topic`.

### Address contract

`host` and `port` are the two halves of one websocket address, `ws://<host>:<port>`,
and each is graded twice: by the domain every dialled address in the package
shares, then by this transport's own narrower rule. For the port that is the
16-bit space followed by autobahn's ceiling of 65534; for the host it is "a string
a websocket URI can carry" followed by the bare-hostname allowlist, which is
stricter than the shared domain and refuses spellings a URI would accept (a
bracketed IPv6 literal, for one). Either half being unusable is reported the same
way whichever it is - `use_rosbridge` returns an error result and `RosbridgeRobot`
raises `ValueError` - and both are refused before a socket is dialled, so the
caller learns the same thing whether or not `roslibpy` is installed.

### Drive contract

`drive()` takes the fleet-standard `(linear, angular, duration, count)` shape,
and three of its guarantees are fleet-standard too: every value is checked
against the same numeric domains the [ROS 2](../ros2-integration.md) and
[RTPS](rtps-robot.md) bridges use, a bare single-shot command latches, and
a timed command is followed by a trailing zero Twist. The velocity clamps and the
`max_duration` ceiling are shared with the Ackermann car bridge
(`AckermannRosRobot`, which declares a `max_speed` and a `max_duration` of its
own) but not with those two, which accept no ceilings because neither knows the
limits of the robot it drives and put the requested value on the wire
unclamped.

```python
# Direct, programmatic control:
robot.drive(linear=1.0, angular=0.0, duration=2.0)  # hold for 2 seconds
robot.drive(linear=1.0)  # latch until stop() - single-shot
robot.stop()  # publishes zero Twist; gated like any cmd_vel publish
pose = robot.get_pose()  # read one odometry sample
scan = robot.get_scan()  # read one laser scan (error if no scan_topic)
```

**Safety semantics** (fleet-wide unless marked):

- **Finite-input guards**: non-finite (NaN, inf) linear or angular velocities
  are rejected before any publish.
- **Single-shot latch**: a bare single-message `drive()` (no `duration`, no
  `count > 1`) publishes once and latches in the robot's controller until
  `stop()` is called. This is standard cmd_vel behavior.
- **Velocity clamps** (not on every mobile base): linear and angular are
  independently clamped to `max_linear` and `max_angular`. `AckermannRosRobot`
  clamps to a `max_speed` of its own the same way. The ROS 2 and RTPS bridges
  accept no velocity ceiling and put the requested value on the wire unchanged.
- **Loud duration rejection** (not on every mobile base): `duration` must be
  positive, finite, and at most `max_duration`; anything else returns a detailed
  error and nothing is published. `AckermannRosRobot` refuses a longer hold
  against a `max_duration` of its own, which is lower, so a hold this bridge
  accepts can be refused on that car. The ROS 2 and RTPS bridges accept any
  positive finite hold.
- **Timed-command trailing zero**: every drive with a `duration` argument and a
  non-zero command (or multi-message publish) automatically publishes a single
  zero Twist afterwards - even if the main publish failed - so a timed drive
  does not leave the robot with a live velocity. That zero is itself a gated
  command, so when it is the call that fails - a declined approval, a rate limit,
  a transport error - `drive` returns that failure instead of the hold's success,
  naming the still-live topic and telling you to halt with `stop`. This was
  previously this bridge's alone; the ROS 2 and RTPS bridges inherit it from the
  shared mobile base now, so a timed drive self-stops on all three.
- **stop() needs no prior state**: the stop method publishes a zero Twist
  regardless of whether an earlier command succeeded, and there is no enable
  handshake to satisfy first. It is *not* exempt from the operator gate: that
  gate is keyed on the command surface rather than on the payload, so a halt to
  a blocklisted `cmd_vel` is approved exactly like a full-speed drive on every
  mobile base, this one included. Pre-approve the topic with
  `STRANDS_ROS2_COMMAND_ALLOW` if the halt must go out unattended.

| Method | ROS action | Notes |
|--------|------------|-------|
| `drive(linear, angular, duration=, count=)` | publish `Twist` to `cmd_vel_topic` | `duration` holds the command at `publish_rate` Hz; no `duration` latches until stop |
| `stop()` | publish zero `Twist` | needs no prior state; gated on the surface like `drive()` |
| `get_pose(timeout=5.0)` | echo `odom_topic` | returns up to 1 sample; a non-positive or non-finite `timeout` is refused by name |
| `get_scan(timeout=5.0)` | echo `scan_topic` | error when no `scan_topic` configured; same `timeout` domain |
| `.tools` | - | per-instance named agent tools |

### from_curiosity

The NASA Curiosity rover Gazebo simulation (ROS 1 Noetic) is pre-wired:

```python
rover = RosbridgeRobot.from_curiosity(
    host="localhost",  # or Docker container IP / another machine
    port=9090,
    node_name="curiosity",  # optional
    # ... any constructor param as override
)
```

The stock wiring uses the rover's `ackermann_drive_controller` (consumes
`geometry_msgs/Twist` directly - no kinematic model needed on the agent side),
and imports safety limits (max_linear=2.0 m/s, max_angular=1.0 rad/s,
max_duration=30s) ported from the strands-robots-ros2 registry.

## Curiosity quickstart (Docker, headless)

Build a self-contained ROS 1 Noetic + Gazebo + rosbridge image, then launch
the agent on your machine:

```bash
# 1. Sim container: ROS1 Noetic + Gazebo (server only) + rosbridge
docker build -t curiosity-sim - <<'EOF'
FROM ros:noetic-robot
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins ros-noetic-controller-manager \
    ros-noetic-joint-state-controller ros-noetic-effort-controllers \
    ros-noetic-position-controllers ros-noetic-velocity-controllers \
    ros-noetic-robot-state-publisher ros-noetic-xacro \
    ros-noetic-rosbridge-suite git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/mark-gl/curiosity_mars_rover_ws.git /ws
WORKDIR /ws
RUN bash -lc "source /opt/ros/noetic/setup.bash && catkin_make \
    -DCATKIN_WHITELIST_PACKAGES='ackermann_drive_controller;curiosity_mars_rover_description;curiosity_mars_rover_control;curiosity_mars_rover_gazebo'"
CMD bash -lc "source /ws/devel/setup.bash && \
    roslaunch curiosity_mars_rover_gazebo main_mars_terrain.launch gui:=false rviz:=false & \
    sleep 30 && roslaunch rosbridge_server rosbridge_websocket.launch"
EOF
docker run -d --name curiosity -p 9090:9090 curiosity-sim

# 2. Agent side (this machine - no ROS needed)
pip install "strands-robots[rosbridge]" strands-agents
python examples/rosbridge/curiosity_agent.py
```

The simulation takes ~30 seconds to warm up. Then the agent will read the
rover's initial pose, execute two drive legs with a turn, and report the
displacement.
