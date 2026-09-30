---
title: fanuc_m710ic
description: "fanuc_m710ic (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# fanuc_m710ic (arm URDF from robot_descriptions)

{{robot_chips:fanuc_m710ic}}

URDF from [robot-descriptions/fanuc_m710ic_description@d12af44](https://github.com/robot-descriptions/fanuc_m710ic_description/tree/d12af44559cd7e46f7afd513237f159f82f8402e), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/fanuc_m710ic.webp" alt="fanuc_m710ic, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("fanuc_m710ic")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("fanuc_m710ic"))
robot.cleanup()
```

Add a cube and a camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
