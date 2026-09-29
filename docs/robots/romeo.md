---
title: romeo
description: "romeo (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# romeo (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">61 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [ros-aldebaran/romeo_robot](https://github.com/ros-aldebaran/romeo_robot/tree/0.1.5), compiled for MuJoCo by `strands_robots` on first use: 61 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/romeo.webp" alt="romeo rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("romeo")
```
