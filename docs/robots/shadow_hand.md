---
title: shadow_hand
description: "Shadow Dexterous Hand (24-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Shadow Dexterous Hand (24-DOF)

{{robot_chips:shadow_hand}}

<robot-viewer name="shadow_hand"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("shadow_hand")
print(robot.robot_action_keys("shadow_hand"))
robot.cleanup()
```

`robot_action_keys` lists the actuators a policy drives; command them by name with `send_action` or from a [policy](../learn/policies/index.md).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/shadow_hand](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/shadow_hand), scene `scene_left.xml`.
