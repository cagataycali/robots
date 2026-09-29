---
title: trifinger_edu
description: "trifinger_edu (educational URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# trifinger_edu (educational URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">9 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [facebookresearch/differentiable-robot-model](https://github.com/facebookresearch/differentiable-robot-model/tree/d7bd1b3b8ef1d6dabe9b68474a622185c510e112), compiled for MuJoCo by `strands_robots` on first use: 9 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/trifinger_edu.webp" alt="trifinger_edu rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("trifinger_edu")
```
