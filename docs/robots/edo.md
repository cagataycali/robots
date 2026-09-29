---
title: edo
description: "edo (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# edo (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [ianathompson/eDO_description@17b3f92](https://github.com/ianathompson/eDO_description/tree/17b3f92f834746106d6a4befaab8eeab3ac248e6), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/edo.webp" alt="edo, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("edo")
```
