---
title: asimov_v0
description: "Asimov V0 Bipedal Legs (12-DOF + 2 passive toes)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Asimov V0 Bipedal Legs (12-DOF + 2 passive toes)

{{robot_chips:asimov_v0}}

<robot-viewer name="asimov_v0"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("asimov_v0", position=[0.0, 0.0, 0.0047])
print(robot.robot_joint_names("asimov_v0"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `asimov`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [menloresearch/asimov-v0/sim-model](https://github.com/menloresearch/asimov-v0/tree/759204f531b071e65540e8b31d89736a3d09e0dd/sim-model), scene `xmls/asimov.xml`.
