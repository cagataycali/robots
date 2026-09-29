---
title: valkyrie
description: "valkyrie (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# valkyrie (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">59 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [gkjohnson/nasa-urdf-robots](https://github.com/gkjohnson/nasa-urdf-robots/tree/54cdeb1dbfb529b79ae3185a53e24fce26e1b74b), compiled for MuJoCo by `strands_robots` on first use: 59 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/valkyrie.webp" alt="valkyrie rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("valkyrie")
```
