---
description: Five Strands @tool helpers for hardware bring-up — calibrate, camera, teleop, pose, serial.
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
    gr00t_inference,        # (covered on the GR00T page)
    robot_mesh,             # (covered on the multi-robot page)
)

ports = serial_tool(action="list")["ports"]
cams  = lerobot_camera(action="list")["cameras"]
```

Each function returns a Strands tool dict (`{"status": ..., "content": [...]}`),
making them droppable into `Agent(tools=[...])`.

## `lerobot_calibrate`

Run the LeRobot calibration walkthrough for a real arm — joint zero, mid pose, limits.
Saves a JSON calibration file LeRobot picks up on subsequent loads.

```python
from strands_robots.tools import lerobot_calibrate

lerobot_calibrate(
    robot_type="so100",
    port="/dev/tty.usbserial-A50285BI",
    calibration_dir="~/.cache/lerobot/calibration/so100",
)
```

| Param | What |
|-------|------|
| `robot_type` | LeRobot robot type string. |
| `port` | Serial device path. |
| `calibration_dir` | Where to save the JSON. Default `~/.cache/lerobot/calibration/{type}`. |

## `lerobot_camera`

Bring-up helper for cameras: list, test, and capture frames.

```python
from strands_robots.tools import lerobot_camera

# List connected cameras across known backends
cams = lerobot_camera(action="list")["cameras"]

# Capture a single frame to disk
lerobot_camera(action="test", index=0, output="test_frame.png")

# Stream for a few seconds
lerobot_camera(action="stream", index=0, duration=5.0, fps=30)
```

The action surface is `list`, `test`, `stream`, plus per-backend specifics. Useful
during the camera-mapping step of bring-up before `Robot(mode="real")`.

## `lerobot_teleoperate`

Run a leader-follower teleoperation loop locally.

```python
from strands_robots.tools import lerobot_teleoperate

lerobot_teleoperate(
    leader_port="/dev/tty.usbserial-LEADER",
    follower_port="/dev/tty.usbserial-FOLLOWER",
    fps=30,
    duration=60.0,    # seconds; omit for unbounded
)
```

For cross-machine teleop (leader and follower on different hosts), use
`InputPublisher` / `InputReceiver` from `strands_robots.mesh` instead — see
[Tutorial 5 — Multi-robot](../tutorial/05-multi-robot.md), step 7.

## `pose_tool`

End-effector pose helpers: forward / inverse kinematics, gripper open/close.

```python
from strands_robots.tools import pose_tool

# Forward kinematics — joints → end-effector pose
pose = pose_tool(action="fk", joint_positions=[0.0]*6, robot_type="so100")["pose"]

# Inverse kinematics — pose → joint positions
joints = pose_tool(action="ik", pose=[0.3, 0.0, 0.2, 0, 0, 0, 1])["joint_positions"]

# Gripper state
pose_tool(action="set_gripper", state="open")
```

Useful for scripted interventions ("move to this exact pose") without a full policy.

## `serial_tool`

Serial port enumeration and basic talk-to-device:

```python
from strands_robots.tools import serial_tool

# Discover serial ports
ports = serial_tool(action="list")["ports"]

# Send a command
result = serial_tool(action="send", port="/dev/tty.usbserial-A50285BI",
                     baud=1000000, command="ping\n")
```

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

- [Robot control](robot-control.md) — the `HardwareRobot` class these tools support.
- [Tutorial 8 — Real hardware](../tutorial/08-real-hardware.md) — when each tool runs.
- [GR00T](../policies/groot.md) — the `gr00t_inference` tool's container lifecycle.
- [Multi-robot](../tutorial/05-multi-robot.md) — the `robot_mesh` tool.
