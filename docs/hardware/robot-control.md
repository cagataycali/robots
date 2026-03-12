# Robot Control

Controlling real hardware with Strands Robots. Same code as simulation, real consequences.

---

## Quick Start

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100", mode="real", cameras={
    "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}
}, port="/dev/ttyACM0")

agent = Agent(tools=[robot])
agent("Pick up the red cube using the mock policy")
```

---

## Auto-Detection

`Robot("so100")` probes USB for Feetech/Dynamixel servo controllers. If found, it creates a `HardwareRobot` automatically:

```python
robot = Robot("so100")  # Auto-detects USB hardware
```

Force real mode explicitly:

```python
robot = Robot("so100", mode="real", cameras={...})
```

Or via environment variable:

```bash
export STRANDS_ROBOT_MODE=real
```

---

## HardwareRobot Actions

The `HardwareRobot` is a Strands `AgentTool`. Agents call it with these actions:

| Action | Description |
|---|---|
| `execute` | Run instruction with policy (blocking) |
| `start` | Start async task execution (non-blocking) |
| `status` | Get current task status |
| `stop` | Stop running task |
| `record` | Execute + record to LeRobotDataset |
| `replay` | Replay a recorded episode on hardware |
| `features` | Show robot observation/action features |

---

## Policy Providers for Hardware

```python
# GR00T on Jetson (ZMQ)
agent("Pick up the cube using groot policy on port 5555")

# LeRobot local inference (ACT)
agent("Pick up the cube using lerobot_local with model lerobot/act_aloha_sim")

# LeRobot async (gRPC server)
agent("Pick up the cube using lerobot_async on port 8080")

# Mock policy (testing)
agent("Test the robot with the mock policy")
```

---

## Camera Configuration

```python
robot = Robot("so100", mode="real", cameras={
    "wrist": {
        "type": "opencv",
        "index_or_path": "/dev/video0",
        "fps": 30,
        "width": 640,
        "height": 480,
        "rotation": 0,
        "color_mode": "rgb",
    },
    "overhead": {
        "type": "opencv",
        "index_or_path": "/dev/video2",
        "fps": 30,
    }
})
```

---

## Control Frequency

Default is 50Hz. Adjust for your robot:

```python
robot = Robot("so100", mode="real",
    control_frequency=30.0,  # 30Hz control loop
    cameras={...},
)
```

---

## Calibration

Robots must be calibrated before use. Use LeRobot's calibration tools:

```bash
# Via LeRobot CLI
lerobot-calibrate --robot so100_follower

# Via the lerobot_calibrate tool
from strands_robots.tools.lerobot_calibrate import lerobot_calibrate
agent = Agent(tools=[lerobot_calibrate])
agent("Calibrate the SO-100 robot")
```
