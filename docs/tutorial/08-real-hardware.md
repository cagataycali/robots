---
description: Sim → real with mode='real'. Calibration, cameras, teleop, safety defaults.
---

# 8 — Real hardware

Same `Robot()` factory call, one extra kwarg, and you're driving real servos.

This chapter is the **bring-up checklist** for a real arm. Every step is non-negotiable
the first time; subsequent runs skip the calibration if the file is still on disk.

## TL;DR

```python
from strands_robots import Robot

robot = Robot(
    "so100",
    mode="real",
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}},
)

# Same Simulation-style action surface — works with the agent
robot.run_policy(
    instruction="pick up the cube",
    policy_provider="lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    duration=15.0,
)
```

The arm must be powered, USB-connected, and calibrated.

## Setup

```bash
pip install "strands-robots[lerobot]"
```

`[lerobot]` pulls in the LeRobot base library, which provides the underlying servo
drivers (Dynamixel, Feetech, etc.). On Linux, you'll also want udev rules for stable
device names — see the [LeRobot docs](https://huggingface.co/docs/lerobot).

## Step 1 — confirm the connection

Plug the controller in, check it shows up:

```bash
ls /dev/tty*       # Linux/macOS — look for tty.usbserial-* or ttyUSB*
```

You can also use the included serial tool:

```python
from strands_robots.tools import serial_tool

ports = serial_tool(action="list")["ports"]
print(ports)
# ['/dev/tty.usbserial-A50285BI', '/dev/cu.Bluetooth-Incoming-Port']
```

## Step 2 — calibrate

Calibration writes joint zero offsets and limits to `~/.cache/lerobot/calibration/`.
Run it once per arm; subsequent runs reuse the file.

Use the bundled tool:

```python
from strands_robots.tools import lerobot_calibrate

lerobot_calibrate(
    robot_type="so100",
    port="/dev/tty.usbserial-A50285BI",
    calibration_dir="~/.cache/lerobot/calibration/so100",
)
```

The tool walks you through positioning the arm at the zero pose, mid pose, and limits.
It saves a JSON calibration file LeRobot will pick up on subsequent loads.

## Step 3 — wire up cameras

Real-hardware cameras are passed through the `cameras` kwarg. Each entry is a
`{name: config}` pair:

```python
cameras = {
    "wrist": {
        "type": "opencv",
        "index_or_path": "/dev/video0",
        "fps": 30,
        "width": 640,
        "height": 480,
    },
    "front": {
        "type": "realsense",
        "serial": "123456789",
        "fps": 30,
    },
}
```

Supported camera backends (check the LeRobot version you have installed):

- `opencv` — any V4L / AVFoundation device.
- `realsense` — Intel RealSense via `pyrealsense2`.
- `intel_rs_d400` — D-series RealSense.

For an interactive bring-up, use the camera tool:

```python
from strands_robots.tools import lerobot_camera

# List connected cameras
cams = lerobot_camera(action="list")["cameras"]

# Test a single camera (saves a frame)
lerobot_camera(action="test", index=0, output="test_frame.png")
```

## Step 4 — instantiate

```python
from strands_robots import Robot

robot = Robot(
    "so100",
    mode="real",
    cameras=cameras,
    port="/dev/tty.usbserial-A50285BI",
)
```

`Robot()` validates the calibration, opens the camera streams, and connects to the
servo controller. If anything fails (port busy, calibration missing, camera unplugged),
it raises before any motion.

## Step 5 — first motion

Always start with a slow, no-policy motion to verify joint mapping:

```python
robot.set_joint_positions([0, 0, 0, 0, 0, 0], duration=3.0)  # home
```

If the arm goes somewhere weird, the calibration is off — re-run step 2.

## Step 6 — run a policy

```python
# A local LeRobot checkpoint
robot.run_policy(
    instruction="pick up the cube",
    policy_provider="lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    duration=15.0,
)

# Or GR00T over the network
robot.run_policy(
    instruction="pick up the cube",
    policy_provider="groot",
    server_address="localhost:5555",
    duration=15.0,
)
```

The `Simulation`-style action surface is preserved on `HardwareRobot`; everything you
learned in chapters 2–4 still applies.

## Step 7 — teleoperate

Two patterns:

**Local teleop** (leader and follower on the same machine):

```python
from strands_robots.tools import lerobot_teleoperate

lerobot_teleoperate(
    leader_port="/dev/tty.usbserial-LEADER",
    follower_port="/dev/tty.usbserial-FOLLOWER",
    fps=30,
)
```

**Mesh teleop** (leader on machine A, follower on machine B):

See [Tutorial 5 — Multi-robot](05-multi-robot.md), step 7.

## Safety defaults

Real hardware has a few safety knobs the factory enforces:

- **`mode="real"` is opt-in.** Default is `"sim"` so a fresh script never accidentally
  drives servos.
- **Velocity limits.** The sim/real action specs match, including per-joint velocity
  caps that come from the calibration file.
- **Watchdog.** A control-loop timeout (configurable via `control_frequency=`) returns
  the arm to a held pose if the policy stalls.
- **`emergency_stop()`.** Available via the mesh and via `robot.stop_task()` directly.
- **STRANDS_TRUST_REMOTE_CODE.** `LerobotLocalPolicy` won't run an HF model that
  requires `trust_remote_code=True` until you set the env var. (Important on real
  hardware where the model can move servos.)

## Common gotchas

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Arm twitches at startup | Calibration missing/stale | Re-run `lerobot_calibrate` |
| Camera frames are black | Wrong `index_or_path` | `lerobot_camera(action="list")` |
| Servo error mid-rollout | Velocity limit too tight | Bump `control_frequency` or relax limits in calibration |
| Policy runs in sim, fails on real | Action spec mismatch | Confirm `data_config` matches between record + infer |
| `PermissionError: /dev/ttyUSB0` | Linux permissions | Add user to `dialout` group |

## Recap

- `mode="real"` flips the factory to `HardwareRobot`. Same agent code works.
- Calibration is one-time per arm; cameras are kwargs; safety is on by default.
- Use `lerobot_calibrate`, `lerobot_camera`, `lerobot_teleoperate`, `serial_tool` for
  bring-up.
- Real-hardware policies are the same `create_policy(...)` calls as simulation.

## See also

- [Hardware tools](../hardware/tools.md) — every `tools/*.py` helper with parameters.
- [Hardware robot reference](../hardware/robot-control.md) — the
  `strands_robots.hardware_robot.Robot` class.
- [Tutorial 5 — Multi-robot](05-multi-robot.md) — mesh teleop and remote control.
- [Troubleshooting](../troubleshooting.md) — error → fix table.
