---
title: double_pendulum
description: "double_pendulum (educational URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# double_pendulum (educational URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">2 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [Gepetto/example-robot-data](https://github.com/Gepetto/example-robot-data/tree/d0d9098d752014aec3725b07766962acf06c5418), compiled for MuJoCo by `strands_robots` on first use: 2 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/double_pendulum.webp" alt="double_pendulum rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("double_pendulum")
```
