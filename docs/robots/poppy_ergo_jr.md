---
title: poppy_ergo_jr
description: "poppy_ergo_jr (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# poppy_ergo_jr (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [poppy-project/poppy_ergo_jr_description](https://github.com/poppy-project/poppy_ergo_jr_description/tree/7eb32bd385afa11dea5e6a6b6a4a86a0243aaa2b), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/poppy_ergo_jr.webp" alt="poppy_ergo_jr rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("poppy_ergo_jr")
```
