---
title: simple_humanoid
description: "simple_humanoid (educational humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# simple_humanoid (educational humanoid URDF from robot_descriptions)

{{robot_chips:simple_humanoid}}

URDF from [laas/simple_humanoid_description@4e859ae](https://github.com/laas/simple_humanoid_description/tree/4e859aed7df3c29954c9cca2a1ecb94069f7cfce), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 29 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/simple_humanoid.webp" alt="simple_humanoid, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("simple_humanoid")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("simple_humanoid"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
