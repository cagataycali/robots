---
description: use_ros - bridge a Strands agent to any ROS 2 graph (topics, services) over native rclpy or a docker container, with dynamic message-type resolution.
---

# ROS 2 integration

`use_ros` gives a Strands agent one structured entry point into any ROS 2 graph
on the host or LAN - listing and echoing topics, publishing messages, and
calling services - without shelling out to the `ros2` CLI by hand or
hard-coding message types.

```python
from strands import Agent
from strands_robots.tools import use_ros

agent = Agent(tools=[use_ros])
agent("list the ROS 2 topics, then drive /turtle1 forward and confirm its pose changed")
```

## Backends

The backend is auto-detected; override it with the `ROS2_MODE` environment
variable (`native` | `docker` | `none`).

| Mode | When | How it runs |
|------|------|-------------|
| `native` | `rclpy` is importable in this interpreter | Runs the `ros2` CLI and small in-process `rclpy` helpers directly |
| `docker` | No host ROS 2, but a container has it sourced | Forwards every command via `docker exec` into `ROS2_DOCKER_CONTAINER` (default `ros-dev`) |
| `none` | Neither is available | Every action returns a clear error naming the `[ros2]` extra and the docker fallback |

The ROS 2 client libraries (`rclpy`, `rosidl_runtime_py`) are not on PyPI - they
ship with a system ROS 2 install (apt / RoboStack / the official docker images).
The docker backend needs nothing installed on the host: point a container at
your DDS domain and go.

```bash
# Dev loop with zero host install (macOS, Jetson, CI):
docker run -d --name ros-dev --net host ros:jazzy tail -f /dev/null
export ROS2_MODE=docker ROS2_DOCKER_CONTAINER=ros-dev
```

Relevant environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROS2_MODE` | auto | Force `native`, `docker`, or `none` |
| `ROS2_DOCKER_CONTAINER` | `ros-dev` | Container name for the docker backend |
| `ROS2_DOCKER_SETUP` | `/opt/ros/jazzy/setup.bash` | Sourced inside the container before each command |

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | Active backend (and container name in docker mode) |
| `list_topics` | - | Topics with their message types |
| `list_nodes` | - | Node names |
| `list_services` | - | Services with their types |
| `info` | `topic` or `service` | Topic/node/service details |
| `echo` | `topic` (type auto-resolved) | N samples as JSON |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `service_call` | `service`, `type` | Service response as JSON |
| `exec_raw` | `command` | Output of an arbitrary `ros2 <args>` command |

Message and service types are resolved dynamically through `rosidl_runtime_py`,
so any interface installed in the ROS 2 environment works with no static
registry. Field payloads are plain JSON dicts applied with `set_message_fields`
(the standard ROS 2 idiom); booleans and `null` are preserved.

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
allowlist before reaching the subprocess layer (alphanumerics plus `_ / ~ {}`
for names; `pkg/msg/Name` or `pkg/srv/Name` for types). The `exec_raw` escape
hatch additionally rejects shell metacharacters. Native mode passes argv without
a shell, and docker mode shlex-quotes every token, so neither path interpolates
untrusted strings into a shell.
