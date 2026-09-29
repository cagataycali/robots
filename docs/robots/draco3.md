---
title: draco3
description: "draco3 (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# draco3 (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">27 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [shbang91/draco3_description](https://github.com/shbang91/draco3_description/tree/5afd19733d7b3e9f1135ba93e0aad90ed1a24cc7), compiled for MuJoCo by `strands_robots` on first use: 27 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/draco3.webp" alt="draco3 rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("draco3")
```
