---
title: atlas_v4
description: "atlas_v4 (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# atlas_v4 (humanoid URDF from robot_descriptions)

{{robot_chips:atlas_v4}}

URDF from [openai/roboschool@1.0.49](https://github.com/openai/roboschool/tree/1.0.49), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 30 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/atlas_v4.webp" alt="atlas_v4, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("atlas_v4")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("atlas_v4"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
