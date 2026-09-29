---
title: upkie
description: "upkie (biped wheeled URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# upkie (biped wheeled URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [upkie/upkie_description](https://github.com/upkie/upkie_description/tree/19a91ce69cab6742c613cab104986e3f8a18d6a5), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/upkie.webp" alt="upkie rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("upkie")
```
