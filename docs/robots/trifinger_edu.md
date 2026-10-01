---
title: trifinger_edu
description: "trifinger_edu (educational URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# trifinger_edu (educational URDF from robot_descriptions)

{{robot_chips:trifinger_edu}}

URDF from [facebookresearch/differentiable-robot-model@d7bd1b3](https://github.com/facebookresearch/differentiable-robot-model/tree/d7bd1b3b8ef1d6dabe9b68474a622185c510e112), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 9 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/trifinger_edu.webp" alt="trifinger_edu, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("trifinger_edu")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("trifinger_edu"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
