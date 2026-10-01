---
title: sigmaban
description: "sigmaban (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# sigmaban (humanoid URDF from robot_descriptions)

{{robot_chips:sigmaban}}

URDF from [Rhoban/sigmaban_urdf@d5d023f](https://github.com/Rhoban/sigmaban_urdf/tree/d5d023fd35800d00d7647000bce8602617a4960d), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 20 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/sigmaban.webp" alt="sigmaban, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("sigmaban")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("sigmaban"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
