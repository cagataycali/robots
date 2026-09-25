---
description: use_rtps and RtpsRobot - join a ROS 2 graph as a DDS participant with no rclpy, no sourced ROS distro. Act as a robot over pure RTPS.
---

# Pure-RTPS ROS 2 integration

ROS 2 runs over DDS, and DDS speaks RTPS on the wire. `use_rtps` lets a strands
agent join a ROS 2 graph as a **first-class DDS participant** using only the
pip-installable `cyclonedds` binding - **no `rclpy`, no sourced ROS 2 distro, no
`ros2` CLI**. Because RTPS is stable across ROS 2 distros, one implementation
interoperates with Humble, Jazzy, Rolling, and beyond.

## use_ros vs use_rtps

| | `use_ros` | `use_rtps` |
|---|-----------|------------|
| Role | client / observer | **participant / robot** |
| Backend | in-process `rclpy` | `cyclonedds` (pip wheel; source build on Linux aarch64) |
| Needs sourced ROS 2 | yes | **no** |
| Type coverage | any installed interface | curated IDL bundle |
| Runs on macOS / CI bare | no (needs ROS) | **yes** |

Use `use_ros` when you have ROS 2 sourced and need full type coverage or
services. Use `use_rtps` when you want zero-install interop or to **act as a
robot** - publishing topics a real ROS 2 stack (rviz, nav2, a teleop node) will
consume, indistinguishable from hardware on the wire.

```bash
pip install 'strands-robots[ros2]'   # cyclonedds - a self-contained wheel on macOS / Windows / Linux x86_64
```

### Linux aarch64 (Jetson)

No cyclonedds release publishes a Linux aarch64 wheel (checked against every
release on PyPI), so on a Jetson, a Thor dev kit or a humanoid's onboard PC the
same command resolves to the **sdist**, and its build needs an existing Cyclone
DDS C install pointed at by `CYCLONEDDS_HOME` (plus `python3-dev`). Two ways to
have one:

```bash
# (a) a sourced ROS 2 distro already ships it
sudo apt install ros-$ROS_DISTRO-cyclonedds
CYCLONEDDS_HOME=/opt/ros/$ROS_DISTRO pip install 'strands-robots[ros2]'

# (b) no ROS 2 on the box: build Cyclone DDS from source (ENABLE_TYPELIB stays ON)
git clone https://github.com/eclipse-cyclonedds/cyclonedds
cmake -S cyclonedds -B cyclonedds/build -DCMAKE_INSTALL_PREFIX=$HOME/cyclonedds
cmake --build cyclonedds/build --target install
export CYCLONEDDS_HOME=$HOME/cyclonedds     # keep it set at runtime too - add to ~/.bashrc
pip install 'strands-robots[ros2]'
```

At runtime the binding locates `libddsc` itself, trying a wheel's bundled copy,
then `$CYCLONEDDS_HOME/lib`, then the normal loader path. So keep
`CYCLONEDDS_HOME` exported whenever the install prefix is somewhere the loader
does not already search - route (b)'s `$HOME/cyclonedds`, or
`/opt/ros/$ROS_DISTRO` in a shell that has not sourced the distro. Install to the
default `/usr/local` prefix instead and `ldconfig` finds `libddsc.so.0`, so the
import needs no variable at all. If `CYCLONEDDS_HOME` *is* set it must be
correct: the loader raises `CycloneDDSLoaderException: Failed to load CycloneDDS
library from <CYCLONEDDS_HOME>/lib/libddsc.so` instead of falling back to the
system path, so a stale export breaks an install that would otherwise work.

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | Whether the cyclonedds backend is available |
| `types` | - | The ROS 2 message types in the local IDL bundle |
| `advertise` | `topic`, `type` | Creates a publisher (appear on the graph) |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `subscribe` | `topic`, `type` | Creates a subscription |
| `echo` | `topic`, `type` | Returns the next N samples as JSON |

Scope (v1) is topics only; services and actions need the ROS 2 request/reply-
over-DDS protocol and are a focused follow-up.

## Type coverage

To publish a message you must own its type definition locally, so `use_rtps`
ships a curated IDL bundle (`strands_robots.rtps.idl`) of the common ROS 2
messages, registered under their ROS 2 type strings: the `geometry_msgs`
primitives (`Twist`/`Pose`/...) plus the `sensor_msgs` `JointState` and `Image`
(with their `std_msgs/Header` + `builtin_interfaces/Time` chain) that the
rclpy-free hardware bridge publishes. List them with
`use_rtps(action="types")`. Arbitrary custom messages are out of scope until
cyclonedds-python's dynamic (XTypes) support matures - use `use_ros` (rclpy) for
those.

ROS 2 names are mangled to their DDS form automatically: a topic `/turtle1/cmd_vel`
becomes `rt/turtle1/cmd_vel`, and a type `geometry_msgs/msg/Twist` becomes
`geometry_msgs::msg::dds_::Twist_` - the conventions that make a bare DDS
participant interoperable with real ROS 2 nodes.

### What counts as a topic name

A name only this package accepts still maps to a DDS topic, and nothing reports
the divergence: DDS matches by topic name, so the participant advertises a name
`rclpy` refuses at `create_publisher` and simply never finds a peer. So the rule
is the ROS 2 mapping's own, in full, and it is enforced once - in
`strands_robots.rtps.mangling` as `ROS_TOPIC_RE` plus `MAX_DDS_TOPIC_LENGTH` -
with every seam that gates a caller name reading it rather than restating it.

| Refused | Because |
| --- | --- |
| `cmd_vel` | not absolute; a DDS write has no namespace to resolve against |
| `/a/` | a name must not end with `/` |
| `//bar`, `/a//b` | a name token must not be empty |
| `/1cam`, `/a/2b` | a token must not start with a digit |
| `/a__b` | a name must not contain repeated underscores |
| `/café`, `/bad name` | only ASCII `[A-Za-z0-9_]` and `/` |
| a name whose `rt`-prefixed form exceeds 256 characters | the mapping bounds the **DDS** name, prefix included |

A digit *inside* a token (`/turtle1/cmd_vel`) is legal, and so is a single
leading underscore (`/_hidden/x`, which ROS 2 treats as a hidden topic) - the
rule narrows to the mapping's set, not to something tighter. A refusal names the
one clause the name broke rather than reporting a generic "invalid topic name".

The `joint_states` / `image_raw` topics the hardware bridges publish on are held
to the same rule: `RosTelemetryBase` sanitises a robot or camera name into a
token that `ROS_TOPIC_RE` accepts, so a camera keyed by its device index
(`0`) publishes on `/<robot>/camera_0/image_raw` rather than on a
`/<robot>/0/image_raw` no ROS 2 node can subscribe to.

Message interfaces only. ROS 2 has no single DDS type for a service or an
action: `rosidl` generates one type per constituent message, so
`example_interfaces/srv/AddTwoInts` becomes
`example_interfaces::srv::dds_::AddTwoInts_Request_` and
`..._Response_` on the `rq`/`rr` prefixes rather than `rt`. A `pkg/srv/Name` or
`pkg/action/Name` type is therefore refused, with the types ROS 2 does generate
quoted - an invented `pkg::srv::dds_::AddTwoInts_` would match no participant,
and DDS reports a type mismatch as silence rather than as an error.

## Examples

```python
from strands_robots import use_rtps

use_rtps(action="status")
use_rtps(action="types")

# Act as a robot: advertise then drive a cmd_vel topic a real node consumes.
use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
use_rtps(action="publish", topic="/turtle1/cmd_vel",
         type="geometry_msgs/msg/Twist",
         fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}},
         count=15, rate=10.0)
```

## Safety

Agent-supplied topic and type names are validated before mangling, against the
same rule the mangling applies (see [What counts as a topic
name](#what-counts-as-a-topic-name)); types must be `pkg/msg/Name`. The tool
never constructs a shell command or generates source, so there is no
command-injection or `eval` surface. Backend, type-resolution, and field errors
are returned as structured `{"status": "error"}` results rather than raised.

The numeric options are checked in the same place, ahead of the backend probe, so
a refusal happens before a writer joins the graph and reports identically whether
or not `cyclonedds` is installed. `count` (`publish`, `echo`) must be a positive
integer, and `rate` (`publish`) and `timeout` (`echo`) must be positive finite
numbers - the same accepted domain `use_ros` enforces, so a value publishable
through one transport is publishable through the other. An option the requested
action never reads is not second-guessed.

## See also

- [Act as a ROS 2 robot over pure RTPS](ros2/rtps-robot.md) - `RtpsRobot`, the rclpy-free hardware bridge, and the DDS Security gate
- [ROS 2 integration](ros2-integration.md) - the `use_ros` tool on a sourced distro
