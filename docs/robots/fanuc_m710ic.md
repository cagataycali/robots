---
title: fanuc_m710ic
description: "fanuc_m710ic (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# fanuc_m710ic (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [robot-descriptions/fanuc_m710ic_description](https://github.com/robot-descriptions/fanuc_m710ic_description/tree/d12af44559cd7e46f7afd513237f159f82f8402e), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/fanuc_m710ic.webp" alt="fanuc_m710ic rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("fanuc_m710ic")
```
