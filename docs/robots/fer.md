---
title: fer
description: "fer (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# fer (arm URDF from robot_descriptions)

{{robot_chips:fer}}

URDF from [frankarobotics/franka_description@1aa4fd3](https://github.com/frankarobotics/franka_description/tree/1aa4fd30e6e274cbf5e986a5af8004df32bad284), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 9 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/fer.webp" alt="fer, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("fer")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("fer"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
