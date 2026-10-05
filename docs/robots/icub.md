---
title: icub
description: "icub (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# icub (humanoid URDF from robot_descriptions)

{{robot_chips:icub}}

URDF from [robotology/icub-models@v1.25.0](https://github.com/robotology/icub-models/tree/v1.25.0), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 32 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/icub.webp" alt="icub, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("icub")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("icub"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
