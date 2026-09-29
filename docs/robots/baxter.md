---
title: baxter
description: "baxter (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# baxter (dual_arm URDF from robot_descriptions)

{{robot_chips:baxter}}

URDF from [RethinkRobotics/baxter_common@6c4b0f3](https://github.com/RethinkRobotics/baxter_common/tree/6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 15 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/baxter.webp" alt="baxter, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("baxter")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
