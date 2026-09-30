---
title: unitree_h1
description: "Unitree H1 Humanoid (19-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree H1 Humanoid (19-DOF)

{{robot_chips:unitree_h1}}

<robot-viewer name="unitree_h1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_h1")
print(robot.robot_joint_names("unitree_h1"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `h1`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/unitree_h1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_h1), scene `scene.xml`.
