---
title: jaxon
description: "jaxon (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# jaxon (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">38 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [robot-descriptions/jaxon_description](https://github.com/robot-descriptions/jaxon_description/tree/4a0cb7a4a737864312f8d6e3f89823a741539bfc), compiled for MuJoCo by `strands_robots` on first use: 38 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/jaxon.webp" alt="jaxon rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("jaxon")
```
