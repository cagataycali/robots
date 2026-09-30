---
title: ur10_official
description: "ur10_official (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ur10_official (arm URDF from robot_descriptions)

{{robot_chips:ur10_official}}

URDF from [UniversalRobots/Universal_Robots_ROS2_Description@22f055d](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/22f055da2fa7e2158254426107d1f257fd56aebb), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/ur10_official.webp" alt="ur10_official, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("ur10_official")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
