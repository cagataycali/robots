# Hardware Tools

Strands tools for working with physical robot hardware.

---

## Available Tools

| Tool | Import | Description |
|---|---|---|
| `gr00t_inference` | `from strands_robots import gr00t_inference` | Manage GR00T Docker inference services |
| `lerobot_camera` | `from strands_robots import lerobot_camera` | Discover, capture, record, preview cameras |
| `teleoperator` | `from strands_robots import teleoperator` | Record demonstrations via teleoperation |
| `lerobot_dataset` | `from strands_robots import lerobot_dataset` | Create and manage LeRobot datasets |
| `pose_tool` | `from strands_robots import pose_tool` | Store, load, and execute named robot poses |
| `serial_tool` | `from strands_robots import serial_tool` | Low-level Feetech servo communication |
| `stream` | `from strands_robots import stream` | Real-time telemetry streaming |
| `stereo_depth` | `from strands_robots import stereo_depth` | Stereo depth estimation |
| `robot_mesh` | `from strands_robots import robot_mesh` | Zenoh P2P robot mesh networking |

---

## Camera Discovery

```python
from strands import Agent
from strands_robots import lerobot_camera

agent = Agent(tools=[lerobot_camera])
agent("Discover all connected cameras and capture a test frame from each")
```

---

## Teleoperation

Record demonstrations for training:

```python
from strands import Agent
from strands_robots import teleoperator

agent = Agent(tools=[teleoperator])
agent("Start teleoperation recording for the SO-100")
```

---

## GR00T Inference

Manage GR00T Docker services:

```python
from strands import Agent
from strands_robots import gr00t_inference

agent = Agent(tools=[gr00t_inference])
agent("Start GR00T inference service on port 5555")
```

---

## Pose Management

Save and replay robot poses:

```python
from strands import Agent
from strands_robots import pose_tool

agent = Agent(tools=[pose_tool])
agent("Save the current position as 'home'")
agent("Move to the 'home' pose")
```

---

## Robot Mesh

Peer-to-peer networking between robots:

```python
from strands import Agent
from strands_robots import robot_mesh

agent = Agent(tools=[robot_mesh])
agent("Show all robots on the mesh")
agent("Send emergency stop to all peers")
```

---

## Composing Tools

All tools compose naturally with agents:

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference, lerobot_camera, pose_tool

agent = Agent(tools=[
    Robot("so100"),
    gr00t_inference,
    lerobot_camera,
    pose_tool,
])

agent("Discover cameras, start GR00T on port 5555, then pick up the red block")
```
