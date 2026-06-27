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
from strands_robots.tools import use_ros

agent = Agent(tools=[use_ros])
agent("list the ROS 2 topics, then drive /turtle1 forward and confirm its pose changed")
```

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

# Publish a velocity command
use_ros(action="publish", topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})

# Call a service with a JSON request
use_ros(action="service_call", service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"})
```

## Safety

Agent-supplied topic, service, and type names are validated against an
allowlist before reaching the rclpy graph/type API (alphanumerics plus
`_ / ~ {}` for names; `pkg/msg/Name` or `pkg/srv/Name` for types). Because the
tool never constructs a shell command or generates source, there is no
command-injection or `eval` surface to defend - the validation simply keeps
malformed names from reaching the ROS 2 client library. Backend and timeout
failures are returned as structured `{"status": "error"}` results rather than
raised exceptions.

## Mesh bridge: a ROS 2 robot as a first-class strands Robot

`use_ros` is the low-level surface. For mobile bases that expose the usual
`cmd_vel` / odometry / scan topic trio, `RosBridgedRobot` wraps that wiring so a
remote ROS 2 robot drives like any other strands robot - the same
`Agent(tools=[robot])` pattern used for simulated and hardware arms.

```python
from strands import Agent
from strands_robots.mesh import RosBridgedRobot

turtle = RosBridgedRobot.from_ros(
    node_name="turtlesim",
    cmd_vel_topic="/turtle1/cmd_vel",
    odom_topic="/turtle1/pose",
    odom_type="turtlesim/msg/Pose",  # optional; auto-resolved when omitted
)

# Direct, programmatic control:
turtle.drive(linear=1.0, duration=1.5)   # hold the command for 1.5 s
print(turtle.get_pose())                 # one odom/pose sample
turtle.stop()

# Or hand the robot to an agent - its capabilities become named tools
# (drive_turtlesim, get_pose_turtlesim, ...):
agent = Agent(tools=turtle.tools)
agent("drive forward for two seconds, then tell me the pose")
```

The bridge is intentionally thin: every method forwards to `use_ros`, so it
inherits the same in-process rclpy backend and input validation. Construct it
freely without a ROS 2 environment present - errors surface only when a method
is actually called and `rclpy` is unavailable.

| Method | ROS 2 action | Notes |
|--------|--------------|-------|
| `drive(linear, angular, duration=, count=)` | publish `Twist` to `cmd_vel_topic` | `duration` holds the command at `publish_rate` Hz |
| `stop()` | publish zero `Twist` | |
| `get_pose()` | echo `odom_topic` | |
| `get_scan()` | echo `scan_topic` | error when no `scan_topic` configured |
| `.tools` | - | per-instance named agent tools |

See `examples/ros2/turtlebot_demo.py` for an end-to-end agent driving a turtle
in `turtlesim` through the mesh bridge.

![Agent driving a turtle via the ROS 2 mesh bridge](assets/ros2_mesh_bridge_turtle.gif)

The trail above is a turtle in `turtlesim` driven entirely through
`RosBridgedRobot.drive(...)` - the velocity commands are published over ROS 2 by
the mesh bridge, and the pose is read back through the same bridge.
