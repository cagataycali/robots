---
title: pr2
description: "pr2 (dual_arm mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# pr2 (dual_arm mobile_manipulator URDF from robot_descriptions)

{{robot_chips:pr2}}

URDF from [ankurhanda/robot-assets@12f1a3c](https://github.com/ankurhanda/robot-assets/tree/12f1a3c89c9975194551afaed0dfae1e09fdb27c), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 38 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/pr2.webp" alt="pr2, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("pr2")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("pr2"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
