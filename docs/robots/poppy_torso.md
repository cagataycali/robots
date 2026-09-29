---
title: poppy_torso
description: "poppy_torso (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# poppy_torso (dual_arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip">13 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [poppy-project/poppy_torso_description@6beeec3](https://github.com/poppy-project/poppy_torso_description/tree/6beeec3d76fb72b7548cce7c73aad722f8884522), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 13 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/poppy_torso.webp" alt="poppy_torso, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("poppy_torso")
```
