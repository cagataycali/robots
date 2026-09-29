---
title: poppy_torso
description: "poppy_torso (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# poppy_torso (dual_arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip">13 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [poppy-project/poppy_torso_description](https://github.com/poppy-project/poppy_torso_description/tree/6beeec3d76fb72b7548cce7c73aad722f8884522), compiled for MuJoCo by `strands_robots` on first use: 13 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/poppy_torso.webp" alt="poppy_torso rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("poppy_torso")
```
