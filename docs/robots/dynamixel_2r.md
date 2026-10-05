---
title: dynamixel_2r
description: "Dynamixel 2R Educational Arm (2-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Dynamixel 2R Educational Arm (2-DOF)

{{robot_chips:dynamixel_2r}}

<robot-viewer name="dynamixel_2r"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("dynamixel_2r")
print(robot.robot_action_keys("dynamixel_2r"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/dynamixel_2r](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/dynamixel_2r), scene `scene.xml`.
