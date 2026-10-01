---
title: ergocub
description: "ergocub (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ergocub (humanoid URDF from robot_descriptions)

{{robot_chips:ergocub}}

URDF from [icub-tech-iit/ergocub-software@v0.7.7](https://github.com/icub-tech-iit/ergocub-software/tree/v0.7.7), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 57 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/ergocub.webp" alt="ergocub, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("ergocub")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("ergocub"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
