---
title: atlas_drc
description: "atlas_drc (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# atlas_drc (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">30 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [RobotLocomotion/drake](https://github.com/RobotLocomotion/drake/tree/7abea0556ede980a5077fe1a8cfbae59b57c7c27), compiled for MuJoCo by `strands_robots` on first use: 30 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/atlas_drc.webp" alt="atlas_drc rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("atlas_drc")
```
