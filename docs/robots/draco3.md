---
title: draco3
description: "draco3 (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# draco3 (humanoid URDF from robot_descriptions)

{{robot_chips:draco3}}

URDF from [shbang91/draco3_description@5afd197](https://github.com/shbang91/draco3_description/tree/5afd19733d7b3e9f1135ba93e0aad90ed1a24cc7), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 27 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/draco3.webp" alt="draco3, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("draco3")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("draco3"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
