---
title: op3
description: "ROBOTIS OP3 Humanoid (20-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ROBOTIS OP3 Humanoid (20-DOF)

{{robot_chips:op3}}

<robot-viewer name="op3"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("op3")
print(robot.robot_joint_names("op3"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `robotis_op3`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/robotis_op3](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/robotis_op3), scene `scene.xml`.
