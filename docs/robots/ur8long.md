---
title: ur8long
description: "ur8long (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ur8long (arm URDF from robot_descriptions)

{{robot_chips:ur8long}}

URDF from [UniversalRobots/Universal_Robots_ROS2_Description@22f055d](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/22f055da2fa7e2158254426107d1f257fd56aebb), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/ur8long.webp" alt="ur8long, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("ur8long")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("ur8long"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
