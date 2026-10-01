---
title: atlas_v4
description: "atlas_v4 (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# atlas_v4 (humanoid URDF from robot_descriptions)

{{robot_chips:atlas_v4}}

URDF from [openai/roboschool@1.0.49](https://github.com/openai/roboschool/tree/1.0.49), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 30 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/atlas_v4.webp" alt="atlas_v4, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("atlas_v4")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
