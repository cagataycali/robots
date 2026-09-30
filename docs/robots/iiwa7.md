---
title: iiwa7
description: "iiwa7 (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# iiwa7 (arm URDF from robot_descriptions)

{{robot_chips:iiwa7}}

URDF from [facebookresearch/differentiable-robot-model@d7bd1b3](https://github.com/facebookresearch/differentiable-robot-model/tree/d7bd1b3b8ef1d6dabe9b68474a622185c510e112), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 7 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/iiwa7.webp" alt="iiwa7, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("iiwa7")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("iiwa7"))
robot.cleanup()
```

Add a cube and a camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
