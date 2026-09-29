---
title: gen3_lite
description: "gen3_lite (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# gen3_lite (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">10 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [Kinovarobotics/ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex/tree/8bf203423911446de28a2248ec87380b7eea2f90), compiled for MuJoCo by `strands_robots` on first use: 10 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/gen3_lite.webp" alt="gen3_lite rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("gen3_lite")
```
