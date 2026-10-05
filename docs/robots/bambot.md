---
title: bambot
description: "bambot (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# bambot (mobile_manipulator URDF from robot_descriptions)

{{robot_chips:bambot}}

URDF from [timqian/bambot@04d9026](https://github.com/timqian/bambot/tree/04d902653794f9f72eeabb09ec90a9af8e397c5b), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 15 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/bambot.webp" alt="bambot, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("bambot")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("bambot"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
