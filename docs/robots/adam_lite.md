---
title: adam_lite
description: "PNDbotics Adam Lite Humanoid (26-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# PNDbotics Adam Lite Humanoid (26-DOF)

{{robot_chips:adam_lite}}

<robot-viewer name="adam_lite"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("adam_lite")
print(robot.robot_joint_names("adam_lite"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `pndbotics_adam_lite`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/pndbotics_adam_lite](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/pndbotics_adam_lite), scene `scene.xml`.
