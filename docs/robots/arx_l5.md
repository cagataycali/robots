---
title: arx_l5
description: "ARX L5 (6-DOF lightweight arm)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ARX L5 (6-DOF lightweight arm)

{{robot_chips:arx_l5}}

<robot-viewer name="arx_l5"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("arx_l5")
print(robot.robot_action_keys("arx_l5"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/arx_l5](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/arx_l5), scene `scene.xml`.
