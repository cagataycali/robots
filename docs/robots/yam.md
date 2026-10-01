---
title: yam
description: "i2rt YAM Arm (8-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# i2rt YAM Arm (8-DOF)

{{robot_chips:yam}}

<robot-viewer name="yam"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("yam")
print(robot.robot_joint_names("yam"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

Aliases: `i2rt_yam`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/i2rt_yam](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/i2rt_yam), scene `scene.xml`.
