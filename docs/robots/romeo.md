---
title: romeo
description: "romeo (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# romeo (humanoid URDF from robot_descriptions)

{{robot_chips:romeo}}

URDF from [ros-aldebaran/romeo_robot@0.1.5](https://github.com/ros-aldebaran/romeo_robot/tree/0.1.5), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 61 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/romeo.webp" alt="romeo, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("romeo")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("romeo"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
