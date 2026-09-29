---
title: berkeley_humanoid
description: "berkeley_humanoid (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# berkeley_humanoid (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">12 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [HybridRobotics/berkeley_humanoid_description](https://github.com/HybridRobotics/berkeley_humanoid_description/tree/d0d13d3f81d795480e25ed1910eaf83d5f0a1d0b), compiled for MuJoCo by `strands_robots` on first use: 12 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/berkeley_humanoid.webp" alt="berkeley_humanoid rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("berkeley_humanoid")
```
