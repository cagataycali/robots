---
title: fetch
description: "fetch (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# fetch (mobile_manipulator URDF from robot_descriptions)

{{robot_chips:fetch}}

URDF from [openai/roboschool@1.0.49](https://github.com/openai/roboschool/tree/1.0.49), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 14 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/fetch.webp" alt="fetch, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("fetch")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("fetch"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
