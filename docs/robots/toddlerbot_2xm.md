---
title: toddlerbot_2xm
description: "Toddlerbot 2xM Humanoid (45-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Toddlerbot 2xM Humanoid (45-DOF)

{{robot_chips:toddlerbot_2xm}}

<robot-viewer name="toddlerbot_2xm"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("toddlerbot_2xm")
print(robot.robot_joint_names("toddlerbot_2xm"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/toddlerbot_2xm](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/toddlerbot_2xm), scene `scene.xml`.
