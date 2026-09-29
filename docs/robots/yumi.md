---
title: yumi
description: "yumi (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# yumi (dual_arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip">18 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [ankurhanda/robot-assets@12f1a3c](https://github.com/ankurhanda/robot-assets/tree/12f1a3c89c9975194551afaed0dfae1e09fdb27c), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 18 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/yumi.webp" alt="yumi, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("yumi")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
