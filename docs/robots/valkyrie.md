---
title: valkyrie
description: "valkyrie (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# valkyrie (humanoid URDF from robot_descriptions)

{{robot_chips:valkyrie}}

URDF from [gkjohnson/nasa-urdf-robots@54cdeb1](https://github.com/gkjohnson/nasa-urdf-robots/tree/54cdeb1dbfb529b79ae3185a53e24fce26e1b74b), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 59 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/valkyrie.webp" alt="valkyrie, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("valkyrie")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("valkyrie"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
