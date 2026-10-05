---
title: pepper
description: "pepper (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# pepper (mobile_manipulator URDF from robot_descriptions)

{{robot_chips:pepper}}

URDF from [jrl-umi3218/pepper_description@cd9715b](https://github.com/jrl-umi3218/pepper_description/tree/cd9715bb5df7ad57445d953db7b1924255305944), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 45 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/pepper.webp" alt="pepper, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("pepper")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("pepper"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
