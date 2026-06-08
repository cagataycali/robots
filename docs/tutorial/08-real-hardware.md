---
description: Sim → real with mode='real'. Calibration, cameras, teleop, safety defaults.
---

# 8 — Real hardware

```python
from strands_robots import Robot
import time

robot = Robot(                                                           # requires hardware
    "so100",
    mode="real",
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}},
    port="/dev/tty.usbserial-A50285BI",
)

robot.start_task(instruction="pick up the cube",
                 policy_provider="groot", policy_port=5555, duration=15.0)

while True:
    s = robot.get_task_status()   # IDLE|CONNECTING|RUNNING|COMPLETED|STOPPED|ERROR
    if s in ("COMPLETED", "STOPPED", "ERROR"):
        break
    time.sleep(0.5)

robot.stop_task()
```

```bash
pip install "strands-robots[lerobot]"   # servo drivers + hardware deps
```

!!! warning "HardwareRobot has no run_policy"
    Use `start_task(instruction, policy_port, policy_provider, duration)` → `get_task_status()` → `stop_task()`. An agent drives hardware via the `execute` / `start` / `status` / `stop` tool actions.

## Bring-up steps

### 1 — Check connection

```python
from strands_robots.tools import serial_tool
print(serial_tool(action="list")["ports"])
```

### 2 — Calibrate (once per arm)

```python
from strands_robots.tools import lerobot_calibrate
lerobot_calibrate(robot_type="so100", port="/dev/tty.usbserial-A50285BI",
                  calibration_dir="~/.cache/lerobot/calibration/so100")
```

Writes joint zeros and limits to `~/.cache/lerobot/calibration/`. Reused on subsequent loads.

### 3 — Verify cameras

```python
from strands_robots.tools import lerobot_camera
lerobot_camera(action="list")                                # list connected cameras
lerobot_camera(action="test", index=0, output="frame.png")  # save a test frame
```

Supported backends: `opencv` (V4L/AVFoundation), `realsense` (pyrealsense2).

### 4 — First motion (sanity check)

```python
robot.set_joint_positions([0, 0, 0, 0, 0, 0], duration=3.0)  # home; re-calibrate if wrong
```

## Mesh teleop

```python
# Machine A — leader publishes at 50 Hz  # requires hardware
leader = Robot("so100", mode="real")
leader.start_teleop_publish(teleoperator=leader.teleoperator,
                            device_name="leader", method="arm", hz=50)

# Machine B — follower applies incoming actions  # requires hardware
follower = Robot("so100", mode="real")
follower.start_teleop_receive(source_peer_id=leader.mesh.peer_id,
                              device_name="leader", apply_fn=None)
leader.stop_teleop("leader"); follower.stop_teleop("leader")
```

Local teleop (same machine): `lerobot_teleoperate(leader_port=..., follower_port=..., fps=30)`.

## Common gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| Arm twitches at startup | Calibration missing/stale | Re-run `lerobot_calibrate` |
| Camera frames are black | Wrong `index_or_path` | `lerobot_camera(action="list")` |
| `PermissionError: /dev/ttyUSB0` | Linux permissions | Add user to `dialout` group |
| `start_task` never reaches RUNNING | Policy server not up | Verify server on specified port |
| Policy runs in sim, fails on real | `data_config` mismatch | Confirm same config at record + infer |

## See also

- [Hardware tools](../hardware/tools.md) — every `tools/*.py` helper.
- [Hardware robot reference](../hardware/robot-control.md) — `hardware_robot.Robot` class.
- [Tutorial 5 — Multi-robot](05-multi-robot.md) — mesh teleop, remote control.
- [Troubleshooting](../troubleshooting.md) — error → fix table.
