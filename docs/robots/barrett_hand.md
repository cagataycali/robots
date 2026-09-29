---
title: barrett_hand
description: "barrett_hand (end_effector URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# barrett_hand (end_effector URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="hand">Hands and grippers</span><span class="sr-chip">8 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [jhu-lcsr-attic/bhand_model](https://github.com/jhu-lcsr-attic/bhand_model/tree/937f4186d6458bd682a7dae825fb6f4efe56ec69), compiled for MuJoCo by `strands_robots` on first use: 8 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/barrett_hand.webp" alt="barrett_hand rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("barrett_hand")
```
