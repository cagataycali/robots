---
title: toddlerbot
description: "toddlerbot (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# toddlerbot (humanoid URDF from robot_descriptions)

{{robot_chips:toddlerbot}}

URDF from [hshi74/toddlerbot@067f9dc](https://github.com/hshi74/toddlerbot/tree/067f9dc4f50143e36334877b9395b9c5c29ee30c), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 30 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/toddlerbot.webp" alt="toddlerbot, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("toddlerbot")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("toddlerbot"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
