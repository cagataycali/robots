---
title: spryped
description: "spryped (biped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# spryped (biped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">8 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [bbokser/spryped](https://github.com/bbokser/spryped/tree/f360a6b78667a4d97c86cad465ef8f4c9512462b), compiled for MuJoCo by `strands_robots` on first use: 8 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/spryped.webp" alt="spryped rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("spryped")
```
