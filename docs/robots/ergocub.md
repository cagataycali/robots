---
title: ergocub
description: "ergocub (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ergocub (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">57 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [icub-tech-iit/ergocub-software](https://github.com/icub-tech-iit/ergocub-software/tree/v0.7.7), compiled for MuJoCo by `strands_robots` on first use: 57 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/ergocub.webp" alt="ergocub rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("ergocub")
```
