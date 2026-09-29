---
title: rhea
description: "rhea (biped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# rhea (biped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [G-Levine/rhea_description](https://github.com/G-Levine/rhea_description/tree/1dc0f1abcf51b5d8a8f7ff8a548399ff0df1414f), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/rhea.webp" alt="rhea rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("rhea")
```
