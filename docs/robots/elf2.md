---
title: elf2
description: "BXI Elf2 Humanoid (25-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# BXI Elf2 Humanoid (25-DOF)

{{robot_chips:elf2}}

<robot-viewer name="elf2"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("elf2")
print(robot.robot_joint_names("elf2"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `bxi_elf2`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [bxirobotics/robot_models/elf2_dof25/xml](https://github.com/bxirobotics/robot_models/tree/eabe24ce937f8e633077a163b883e92e8996c36e/elf2_dof25/xml), scene `scene.xml`.
