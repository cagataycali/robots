---
description: Eight Strands @tool helpers for hardware bring-up and operation — calibrate, camera, teleop, pose, serial, gr00t inference, mesh, download assets.
---

# Hardware tools

The `strands_robots/tools/` directory ships small `@tool`-decorated helpers that an
agent (or your code directly) can use during real-hardware bring-up and operation.

## TL;DR

```python
from strands_robots.tools import (
    lerobot_calibrate,
    lerobot_camera,
    lerobot_teleoperate,
    pose_tool,
    serial_tool,
    gr00t_inference,       # covered on the GR00T page
    robot_mesh,            # covered on the multi-robot page
    download_assets,
)

# All tools return a Strands tool dict: {"status": ..., "content": [{"text": "..."}]}
result = serial_tool(action="list")
result = lerobot_camera(action="list", camera_type="opencv")
```

Each function returns a Strands tool dict (`{"status": ..., "content": [{"text": "..."}]}`),
making them droppable into `Agent(tools=[...])`. Do **not** index custom keys like
`result["ports"]` or `result["cameras"]` - parse `result["content"][0]["text"]` instead.

## `lerobot_calibrate`

Run the LeRobot calibration walkthrough for a real arm - joint zero, mid pose, limits.
Saves a JSON calibration file LeRobot picks up on subsequent loads.

```python
from strands_robots.tools import lerobot_calibrate

# List available devices
result = lerobot_calibrate(action="list")
# Parse result["content"][0]["text"] for the device list

# Calibrate a specific device
result = lerobot_calibrate(
    action="calibrate",
    device_type="robot",
    device_model="so100",
)
```

| Param | What |
|-------|------|
| `action` | Operation: `"list"`, `"calibrate"`, etc. |
| `device_type` | LeRobot device type (e.g. `"robot"`, `"teleoperator"`). |
| `device_model` | Model string (e.g. `"so100"`, `"koch"`). |

Returns `{"status": ..., "content": [{"text": "..."}]}`.

## `lerobot_camera`

Bring-up helper for cameras: list, test, and capture frames.

```python
from strands_robots.tools import lerobot_camera

# List connected cameras across known backends
result = lerobot_camera(action="list", camera_type="opencv")
# Parse result["content"][0]["text"] for the camera list

# Test a specific camera
result = lerobot_camera(action="test", camera_type="opencv", camera_id=0)

# Stream for a few seconds
result = lerobot_camera(action="stream", camera_type="opencv", camera_id=0)
```

| Param | What |
|-------|------|
| `action` | `"list"`, `"test"`, `"stream"`, or backend-specific. |
| `camera_type` | Backend: `"opencv"` (default), `"realsense"`, etc. |
| `camera_id` | Camera index or path. |

Useful during the camera-mapping step of bring-up before `Robot(mode="real")`.

## `lerobot_teleoperate`

Run a leader-follower teleoperation session.

```python
from strands_robots.tools import lerobot_teleoperate

# Start a named session in the background
result = lerobot_teleoperate(
    action="start",
    session_name="my_teleop",
    background=True,
)

# Stop the session
result = lerobot_teleoperate(action="stop", session_name="my_teleop")
```

| Param | What |
|-------|------|
| `action` | `"start"`, `"stop"`, `"status"`, etc. |
| `session_name` | Identifier for the teleop session. |
| `background` | Run the loop in a background thread (default `True`). |

For cross-machine teleop (leader and follower on different hosts), use
`InputPublisher` / `InputReceiver` from `strands_robots.mesh` instead - see
[Tutorial 5 - Multi-robot](../tutorial/05-multi-robot.md), step 7.

## `pose_tool`

End-effector pose helpers: forward / inverse kinematics, gripper open/close.

```python
from strands_robots.tools import pose_tool

# Forward kinematics
result = pose_tool(action="fk", robot_id="so101_follower", port="/dev/ttyACM0")
# Parse result["content"][0]["text"] for the pose

# Inverse kinematics
result = pose_tool(action="ik", robot_id="so101_follower", port="/dev/ttyACM0")

# Gripper state
result = pose_tool(action="set_gripper", robot_id="so101_follower", port="/dev/ttyACM0", state="open")
```

| Param | What |
|-------|------|
| `action` | `"fk"`, `"ik"`, `"set_gripper"`, etc. |
| `robot_id` | Robot identifier (default `"so101_follower"`). |
| `port` | Serial port (default `"/dev/ttyACM0"`). |

Useful for scripted interventions ("move to this exact pose") without a full policy.

## `serial_tool`

Serial port enumeration and basic talk-to-device:

```python
from strands_robots.tools import serial_tool

# Discover serial ports
result = serial_tool(action="list")
# Parse result["content"][0]["text"] for the port list

# Send a command
result = serial_tool(
    action="send",
    port="/dev/ttyACM0",
    baudrate=1000000,
    command="ping\n",
)
```

| Param | What |
|-------|------|
| `action` | `"list"`, `"send"`, etc. |
| `port` | Serial device path. |
| `baudrate` | Baud rate (default `9600`). |

Mostly used during initial connection debugging.

## Importing in agent code

All tools are lazy-loaded. They're cheap to import but pull their heavy dependencies
(`pyserial`, `psutil`, `pyrealsense2`, etc.) only when called:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import (
    lerobot_calibrate,
    lerobot_camera,
    lerobot_teleoperate,
    pose_tool,
    serial_tool,
)

robot = Robot("so100")
agent = Agent(tools=[
    robot,
    lerobot_calibrate,
    lerobot_camera,
    lerobot_teleoperate,
    pose_tool,
    serial_tool,
])

agent("Find a connected so100 arm, calibrate it, list cameras, and teleop "
      "with the leader on /dev/tty.usbserial-LEADER and follower on "
      "/dev/tty.usbserial-FOLLOWER for 30 seconds")
```

## See also

- [Robot control](robot-control.md) - the `HardwareRobot` class these tools support.
- [Tutorial 8 - Real hardware](../tutorial/08-real-hardware.md) - when each tool runs.
- [GR00T](../policies/groot.md) - the `gr00t_inference` tool's container lifecycle.
- [Multi-robot](../tutorial/05-multi-robot.md) - the `robot_mesh` tool.
