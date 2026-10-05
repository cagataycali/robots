---
title: piper
description: "AgileX Piper (6-DOF + gripper)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# AgileX Piper (6-DOF + gripper)

{{robot_chips:piper}}

<robot-viewer name="piper"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("piper")
print(robot.robot_joint_names("piper"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

Aliases: `agilex_piper`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/agilex_piper), scene `scene.xml`.
