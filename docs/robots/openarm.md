---
title: openarm
description: "Enactic OpenArm (7-DOF, DAMIAO motors, CAN bus)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Enactic OpenArm (7-DOF, DAMIAO motors, CAN bus)

{{robot_chips:openarm}}

<robot-viewer name="openarm"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("openarm")
print(robot.robot_action_keys("openarm"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

```python title="sketch"
robot = Robot("openarm", mode="real", port="/dev/ttyACM0")  # lerobot openarm_follower
```

Aliases: `enactic_openarm`, `open_arm`, `openarm_v1`, `openarm_v10`.

## Hardware

**lerobot.** `Robot("openarm", mode="real")` builds lerobot's `openarm_follower` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [enactic/openarm_mujoco/v1](https://github.com/enactic/openarm_mujoco/tree/cd30dd4c0a97832d1c063bf759514ed18fbe04a5/v1), scene `scene.xml`.
