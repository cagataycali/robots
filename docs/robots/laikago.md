---
title: laikago
description: "laikago (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# laikago (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">12 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/f3300ff1bf0ab9efbea0162717353480d9b05d73), compiled for MuJoCo by `strands_robots` on first use: 12 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/laikago.webp" alt="laikago rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("laikago")
```
