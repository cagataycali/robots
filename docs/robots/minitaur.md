---
title: minitaur
description: "minitaur (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# minitaur (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">16 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3/tree/7dee3436e747958e7088dfdcea0e4ae031ce619e), compiled for MuJoCo by `strands_robots` on first use: 16 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/minitaur.webp" alt="minitaur rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("minitaur")
```
