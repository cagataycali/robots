---
title: toddlerbot
description: "toddlerbot (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# toddlerbot (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">30 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [hshi74/toddlerbot](https://github.com/hshi74/toddlerbot/tree/067f9dc4f50143e36334877b9395b9c5c29ee30c), compiled for MuJoCo by `strands_robots` on first use: 30 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/toddlerbot.webp" alt="toddlerbot rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("toddlerbot")
```
