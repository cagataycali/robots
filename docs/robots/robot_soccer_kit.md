---
title: robot_soccer_kit
description: "Robot Soccer Kit (multi-robot soccer, 65-DOF total)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Robot Soccer Kit (multi-robot soccer, 65-DOF total)

{{robot_chips:robot_soccer_kit}}

<robot-viewer name="robot_soccer_kit"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("robot_soccer_kit")
print(robot.robot_joint_names("robot_soccer_kit"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

Aliases: `rsk`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/robot_soccer_kit](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/robot_soccer_kit), scene `scene.xml`.
