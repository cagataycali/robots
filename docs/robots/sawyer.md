---
title: sawyer
description: "Rethink Robotics Sawyer (7-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Rethink Robotics Sawyer (7-DOF)

{{robot_chips:sawyer}}

<robot-viewer name="sawyer"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("sawyer")
print(robot.robot_action_keys("sawyer"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

Aliases: `rethink_sawyer`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/rethink_robotics_sawyer](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/rethink_robotics_sawyer), scene `scene.xml`.
