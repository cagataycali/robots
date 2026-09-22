---
description: use_ros - bridge a Strands agent to any ROS 2 graph (topics, services) in-process through rclpy, with dynamic message-type resolution.
---

# ROS 2 integration

`use_ros` gives a Strands agent one structured entry point into any ROS 2 graph
reachable from the interpreter - listing and echoing topics, publishing
messages, and calling services - **entirely in-process through `rclpy`**. There
is no `ros2` CLI shelling and no generated-code snippets: every action calls the
ROS 2 client library directly, so message types are real Python classes, errors
are real exceptions, and a single long-lived node/executor is reused across
calls.

```python
from strands import Agent
from strands_robots import use_ros

agent = Agent(tools=[use_ros])
agent("list the ROS 2 topics, then drive /turtle1 forward and confirm its pose changed")
```

![A Strands agent driving a closed-loop square in turtlesim via use_ros](assets/use_ros_agent_square.gif)

*A Strands agent (Claude Opus via Amazon Bedrock) given the `use_ros` tool drives
a real ROS 2 `turtlesim` in a closed-loop square - reading pose, correcting
heading, and re-driving - over 43 in-process `use_ros` calls. See
[`examples/ros2/use_ros/`](https://github.com/strands-labs/robots/tree/main/examples/ros2/use_ros).*

## ROS 2 surfaces at a glance

strands-robots meets ROS 2 from four complementary angles - pick by what you
have and what you want to do:

| Surface | Role | Backend | Needs sourced ROS 2 | Use it to |
|---------|------|---------|---------------------|-----------|
| **`use_ros`** tool | client / observer + commander | in-process `rclpy` | yes | List/echo/publish topics, call services on any ROS 2 graph - full type coverage |
| **`use_rtps`** tool | participant / **act as a robot** | pure `cyclonedds` (pip) | **no** | Join a graph as a DDS peer and publish topics a real stack consumes; works on macOS/CI/Linux x86_64 from the wheel, all distros; Linux aarch64 (Jetson) builds from source - see [rtps integration](rtps-integration.md#linux-aarch64-jetson) |
| **`use_rosbridge`** tool + **`RosbridgeRobot`** | ROS1 / remote robots over a rosbridge WebSocket | pure-pip `roslibpy` | **no** | Drive ROS1 robots (e.g. the NASA Curiosity Gazebo sim) or any remote rosbridge robot from a machine with no ROS install - see [rosbridge integration](rosbridge-integration.md) |
| **`RosBridgedRobot`** | a ROS 2 robot as a strands `Robot` | in-process `rclpy` | yes | `drive()`/`get_pose()` a `cmd_vel`/odom base with the same `Agent(tools=[robot])` UX as sim/hardware |
| **`AckermannRosRobot`** | an Ackermann ROS 2 car as a strands `Robot` | in-process `rclpy` | yes | `drive()`/`get_scan()` a steering-geometry car (AWS DeepRacer servo stack) with bicycle-model conversion and an automatic enable handshake |
| **`SimEngine(ros2_bridge=True)`** | the **simulation as a ROS node** | `rclpy` | yes | Publish a running MuJoCo sim's `joint_states` + camera `image_raw` so rviz/nav2/agents can subscribe |
| **`Robot(ros2_bridge=True)`** | a **real robot as a ROS node** (full duplex) | `rclpy` | yes | Publish a physical arm's live `joint_states` + camera `image_raw` so rviz/nav2/agents subscribe to the hardware, **and** subscribe to `joint_command` to drive the arm - symmetric to the sim bridge, plus an inbound command path the sim does not need |

The `use_ros` tool is documented below; the safety gate, the Ackermann bridge, the sim,
hardware and mesh bridges each have [their own page](#deeper). The
`use_rtps` pure-RTPS path (no rclpy, every ROS 2 distro) is on the
[Pure-RTPS ROS 2](rtps-integration.md) page.

## Requirements

The tool needs `rclpy` and `rosidl_runtime_py` importable in the same
interpreter that runs the agent. These ship with a sourced system ROS 2 distro
and are **not** on PyPI, so they cannot be `pip install`ed and are not pinned in
`pyproject.toml`. Source a ROS 2 environment before launching the agent:

```bash
source /opt/ros/jazzy/setup.bash   # or your distro / RoboStack / conda env
```

When `rclpy` is not importable, every action returns a clear, actionable error
naming the remedy (it never raises). Check the active backend with
`use_ros(action="status")`, which reports either `rclpy (in-process)` or `none`.

The `[ros2]` extra is minimal and optional - it only pulls the pip-installable
`cyclonedds` DDS RMW binding. It does **not** provision ROS 2 by itself; you
still need a real sourced distro.

```bash
pip install 'strands-robots[ros2]'   # optional cyclonedds RMW binding only
```

The binding is a pre-built wheel on macOS, Windows and Linux x86_64. No
cyclonedds release publishes a Linux **aarch64** wheel, so on a Jetson or a
robot's onboard computer the extra resolves to the sdist, which builds against
an existing Cyclone DDS C install (`CYCLONEDDS_HOME`) - the recipe is in
[rtps integration](rtps-integration.md#linux-aarch64-jetson).

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | Whether the in-process rclpy backend is available |
| `list_topics` | - | Topics with their message types |
| `list_nodes` | - | Node names |
| `list_services` | - | Services with their types |
| `info` | `topic` or `service` | Topic (type + pub/sub counts) or service (type) details |
| `echo` | `topic` (type auto-resolved) | N samples as JSON |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `service_call` | `service`, `type` | Service response as JSON |
| `list_actions` | - | Action servers with their types |
| `action_send_goal` | `action_name`, `type` | Terminal `{goal_status, result, feedback}` as JSON; goal is cancelled if `timeout` expires |

Graph introspection (`list_*`, `info`, `echo` type auto-resolution) uses the
rclpy node API directly (`get_topic_names_and_types`, `get_node_names_and_namespaces`,
`get_service_names_and_types`, `count_publishers`/`count_subscribers`). Message
and service types are resolved dynamically through `rosidl_runtime_py`
(`get_message` / `get_service`), so any interface installed in the ROS 2
environment works with no static registry. Field payloads are plain Python
dicts applied with `set_message_fields` (the standard ROS 2 idiom) - passed
straight to rclpy, never serialised through source, so booleans and `null` are
preserved by construction.

## Examples

```python
use_ros(action="status")
use_ros(action="list_topics")

# Subscribe and read two samples (type auto-resolved from the graph)
use_ros(action="echo", topic="/turtle1/pose", count=2, timeout=2.0)

# Publish a velocity command. /cmd_vel is a gated surface - see
# ros2/safety.md#safety-critical-command-surfaces-need-operator-approval.
use_ros(action="publish", topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})

# Call a service with a JSON request
use_ros(action="service_call", service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"})
```

## Try it live

A reproducible, one-command showcase drives a real `turtlesim` through every
`use_ros` action (in-process rclpy, closed sense->act->sense loop), and a second
service runs a Strands Agent that draws the square above from a plain-English
prompt:

```bash
cd examples/ros2/use_ros
docker compose run --build --rm showcase   # every action; exits 0 iff the turtle moved
docker compose run --build --rm agent      # a Strands Agent drives a closed-loop square
```

Captured runs are in `examples/ros2/use_ros/sample_output.txt` and
`agent_sample_output.txt`.

## Deeper

- [Safety: input validation and the command gate](ros2/safety.md) - what is
  validated before a name reaches rclpy, and the operator approval every
  safety-critical command surface needs
- [Ackermann robots (AWS DeepRacer)](ros2/ackermann.md) - `AckermannRosRobot`
- [Sim bridge](ros2/sim-bridge.md) - `SimEngine(ros2_bridge=True)`
- [Hardware bridge](ros2/hardware-bridge.md) - `Robot(ros2_bridge=True)` and
  the inbound `joint_command` surface
- [Mesh bridge](ros2/mesh-bridge.md) - `RosBridgedRobot` and the shared
  mobile-base contract
- [Pure-RTPS ROS 2](rtps-integration.md) - `use_rtps`, no rclpy
- [rosbridge integration](rosbridge-integration.md) - `use_rosbridge`, ROS 1
  and remote robots
