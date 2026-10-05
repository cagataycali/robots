---
title: ginger
description: "ginger (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ginger (mobile_manipulator URDF from robot_descriptions)

{{robot_chips:ginger}}

URDF from [Rayckey/GingerURDF@6a1307c](https://github.com/Rayckey/GingerURDF/tree/6a1307cd0ee2b77c82f8839cdce3a2e2eed2bd8f), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 49 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/ginger.webp" alt="ginger, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("ginger")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("ginger"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
