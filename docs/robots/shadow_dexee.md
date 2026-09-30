---
title: shadow_dexee
description: "Shadow DexEE Dexterous End-Effector (12-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Shadow DexEE Dexterous End-Effector (12-DOF)

{{robot_chips:shadow_dexee}}

<robot-viewer name="shadow_dexee"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("shadow_dexee")
print(robot.robot_joint_names("shadow_dexee"))
robot.cleanup()
```

`robot_joint_names` lists the finger joints a policy drives; set them by name with `set_joint_positions` or from a [policy](../learn/policies/index.md).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/shadow_dexee](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/shadow_dexee), scene `scene.xml`.
