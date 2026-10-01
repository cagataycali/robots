---
title: unitree_a1
description: "Unitree A1 Quadruped"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree A1 Quadruped

{{robot_chips:unitree_a1}}

<robot-viewer name="unitree_a1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_a1")
print(robot.robot_joint_names("unitree_a1"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

Aliases: `a1`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [unitreerobotics/unitree_mujoco/data/a1](https://github.com/unitreerobotics/unitree_mujoco/tree/f3300ff1bf0ab9efbea0162717353480d9b05d73/data/a1), scene `xml/a1.xml`.
