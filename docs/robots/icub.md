---
title: icub
description: "icub (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# icub (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">32 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [robotology/icub-models](https://github.com/robotology/icub-models/tree/v1.25.0), compiled for MuJoCo by `strands_robots` on first use: 32 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/icub.webp" alt="icub rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("icub")
```
