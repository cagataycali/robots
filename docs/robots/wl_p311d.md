---
title: wl_p311d
description: "wl_p311d (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# wl_p311d (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">16 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [limxdynamics/robot-description](https://github.com/limxdynamics/robot-description/tree/a097533372a08298d45af391cbdfc2fd2dc3da6f), compiled for MuJoCo by `strands_robots` on first use: 16 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/wl_p311d.webp" alt="wl_p311d rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("wl_p311d")
```
