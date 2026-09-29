---
title: leap_hand_v1
description: "leap_hand_v1 (end_effector URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# leap_hand_v1 (end_effector URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="hand">Hands and grippers</span><span class="sr-chip">16 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [leap-hand/LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim/tree/150bc3d4b61fd6619193ba5a8ef209f3609ced89), compiled for MuJoCo by `strands_robots` on first use: 16 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/leap_hand_v1.webp" alt="leap_hand_v1 rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("leap_hand_v1")
```
