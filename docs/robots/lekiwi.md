---
title: lekiwi
description: "LeKiwi mobile manipulator (6-DOF arm on 3-omniwheel base, 9 actuators)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# LeKiwi mobile manipulator (6-DOF arm on 3-omniwheel base, 9 actuators)

{{robot_chips:lekiwi}}

<robot-viewer name="lekiwi"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("lekiwi", position=[0.0, 0.0, 0.0346])
print(robot.robot_action_keys("lekiwi"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

```python title="sketch"
robot = Robot("lekiwi", mode="real", driver="lerobot", port="/dev/ttyACM0")  # lerobot lekiwi
robot = Robot("lekiwi", mode="real", port="/dev/ttyACM0")  # FeetechDriver
```

| Observation key | `send_action` label |
|---|---|
| `base_back_wheel_joint` | `base_back_wheel` |
| `base_right_wheel_joint` | `base_right_wheel` |
| `base_left_wheel_joint` | `base_left_wheel` |
| `Rotation` | `shoulder_pan` |
| `Pitch` | `shoulder_lift` |
| `Elbow` | `elbow_flex` |
| `Wrist_Pitch` | `wrist_flex` |
| `Wrist_Roll` | `wrist_roll` |
| `Jaw` | `gripper` |

## Hardware

**lerobot.** `Robot("lekiwi", mode="real", driver="lerobot")` builds lerobot's `lekiwi` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict.

**`FeetechDriver`** (the default for this robot) speaks Feetech STS/SMS serial bus: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#feetechdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [Ekumen-OS/lekiwi/packages/lekiwi_sim/lekiwi_sim/assets](https://github.com/Ekumen-OS/lekiwi/tree/32cf6a69eb320cc22620cdaa529e35f20fc12b1f/packages/lekiwi_sim/lekiwi_sim/assets), scene `scene.xml`.
