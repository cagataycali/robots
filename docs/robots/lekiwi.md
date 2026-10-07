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

robot = Robot("lekiwi")
print(robot.robot_action_keys("lekiwi"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

```python title="sketch"
robot = Robot("lekiwi", mode="real", port="/dev/ttyACM0")  # lerobot lekiwi
```

| Observation key | also accepted by `send_action` |
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

These are `mode="sim"` keys; on hardware `get_observation()` returns lerobot's `<motor>.pos`.

Gripper `Jaw`: the low end closes, the high end opens.

## Hardware

**lerobot.** `Robot("lekiwi", mode="real")` builds lerobot's `lekiwi` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [Ekumen-OS/lekiwi/packages/lekiwi_sim/lekiwi_sim/assets](https://github.com/Ekumen-OS/lekiwi/tree/32cf6a69eb320cc22620cdaa529e35f20fc12b1f/packages/lekiwi_sim/lekiwi_sim/assets), scene `scene.xml`.
