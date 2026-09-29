---
title: baxter
description: "baxter (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# baxter (dual_arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip">15 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [RethinkRobotics/baxter_common](https://github.com/RethinkRobotics/baxter_common/tree/6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e), compiled for MuJoCo by `strands_robots` on first use: 15 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/baxter.webp" alt="baxter rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("baxter")
```
