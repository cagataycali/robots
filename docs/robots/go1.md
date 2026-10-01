---
title: go1
description: "Unitree Go1 Quadruped (12-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree Go1 Quadruped (12-DOF)

{{robot_chips:go1}}

<robot-viewer name="go1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("go1")
print(robot.robot_joint_names("go1"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

Aliases: `unitree_go1`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/unitree_go1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_go1), scene `scene.xml`.
