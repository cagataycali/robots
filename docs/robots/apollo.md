---
title: apollo
description: "Apptronik Apollo Humanoid (34-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Apptronik Apollo Humanoid (34-DOF)

{{robot_chips:apollo}}

<robot-viewer name="apollo"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("apollo")
print(robot.robot_joint_names("apollo"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `apptronik_apollo`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/apptronik_apollo](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/apptronik_apollo), scene `scene.xml`.
