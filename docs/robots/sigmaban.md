---
title: sigmaban
description: "sigmaban (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# sigmaban (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">20 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [Rhoban/sigmaban_urdf](https://github.com/Rhoban/sigmaban_urdf/tree/d5d023fd35800d00d7647000bce8602617a4960d), compiled for MuJoCo by `strands_robots` on first use: 20 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/sigmaban.webp" alt="sigmaban rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("sigmaban")
```
