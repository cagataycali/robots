---
title: jvrc
description: "JVRC-1 Humanoid (HRP-based, 45-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# JVRC-1 Humanoid (HRP-based, 45-DOF)

{{robot_chips:jvrc}}

<robot-viewer name="jvrc"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("jvrc")
print(robot.robot_joint_names("jvrc"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `jvrc1`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [isri-aist/jvrc_mj_description](https://github.com/isri-aist/jvrc_mj_description/tree/0f0ce7daefdd66c54e0909a6bf2c22154844f5f3), scene `xml/jvrc1.xml`.
