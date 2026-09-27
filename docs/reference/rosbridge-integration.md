---
description: use_rosbridge - bridge a Strands agent to any ROS 1 or remote robot over a rosbridge WebSocket (pure pip, no ROS install).
---

# rosbridge integration

`use_rosbridge` speaks the rosbridge JSON protocol over a WebSocket via
pure-pip `roslibpy` - no ROS environment needed on the agent's machine. This
gives rosbridge two properties no other strands-robots transport has:

* **ROS 1 robots** - rosbridge_suite ships for ROS 1 (and ROS 2) alike. Drive
  the NASA Curiosity rover Gazebo simulation (ROS 1 Noetic), or any ROS 1
  system.
* **No ROS install on this machine** - the agent can run on macOS, CI, WSL, or
  any laptop and introspect and drive robots across a network.

```python
from strands import Agent
from strands_robots.mesh import RosbridgeRobot

# Stock NASA-sim wiring: cmd_vel/odom topics and safety limits preconfigured.
rover = RosbridgeRobot.from_curiosity(host="192.168.1.20")
agent = Agent(tools=rover.tools)
agent("drive forward for 5 seconds, then report the odometry")
```

## Requirements

Install the `[rosbridge]` extra:

```bash
pip install "strands-robots[rosbridge]"
```

The robot side runs `rosbridge_server` with `rosapi` enabled - standard in every
rosbridge install. rosbridge is unauthenticated by default: use on trusted
networks only. rosauth is out of scope.

```bash
# ROS 1 Noetic example
apt-get install ros-noetic-rosbridge-suite
roslaunch rosbridge_server rosbridge_websocket.launch

# ROS 2 example
apt-get install ros-<distro>-rosbridge-suite
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | roslibpy availability + connectivity to host:port |
| `list_topics` | - | Every topic rosapi reports, with its type where rosapi reports one (rosapi /rosapi/topics) |
| `list_services` | - | Services (rosapi /rosapi/services) |
| `echo` | `topic` (type auto-resolved) | N samples as JSON |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `service_call` | `service`, `type` | Service response as JSON |

## Examples

```python
from strands_robots import use_rosbridge

# Check connectivity
use_rosbridge(action="status", host="192.168.1.20")

# Graph introspection
use_rosbridge(action="list_topics")
use_rosbridge(action="list_services")

# Subscribe and read one sample (type auto-resolved via rosapi)
use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=1)

# Publish a velocity command
use_rosbridge(action="publish",
              topic="/curiosity_mars_rover/ackermann_drive_controller/cmd_vel",
              type="geometry_msgs/Twist",
              fields={"linear": {"x": 1.0}, "angular": {"z": 0.0}})

# Call a service with a JSON request
use_rosbridge(action="service_call", service="/some_service",
              type="std_srvs/Trigger",
              fields={})
```

Graph introspection uses the `rosapi` node's services. Interface types are
ROS 1-style two-segment names (e.g. `geometry_msgs/Twist`); field payloads are
plain JSON dicts, exactly as rosbridge transmits them.

## RosbridgeRobot

For a mobile base that exposes the standard `cmd_vel` / odometry / scan topic
trio, [`RosbridgeRobot`](ros2/rosbridge-robot.md) wraps that wiring so the robot
drives like any other strands robot, through this same transport. The NASA
Curiosity Docker recipe lives on that page too.

## Security

rosbridge is **unauthenticated by default**. The WebSocket accepts connections
from any network address on the port. For trustworthy deployments:

- **Trusted networks only**: use rosbridge on private intranets, never expose
  port 9090 to the internet.
- **rosauth out of scope**: the `rosauth` ROS package can add authentication,
  but configuration and key distribution are operator responsibilities; this
  library does not provision them.
- **Network isolation**: firewall the rosbridge port to known agent machine IPs.
