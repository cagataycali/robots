---
title: edo
description: "edo (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# edo (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [ianathompson/eDO_description](https://github.com/ianathompson/eDO_description/tree/17b3f92f834746106d6a4befaab8eeab3ac248e6), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/edo.webp" alt="edo rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("edo")
```
