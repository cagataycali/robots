---
title: pr2
description: "pr2 (dual_arm mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# pr2 (dual_arm mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">38 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [ankurhanda/robot-assets](https://github.com/ankurhanda/robot-assets/tree/12f1a3c89c9975194551afaed0dfae1e09fdb27c), compiled for MuJoCo by `strands_robots` on first use: 38 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/pr2.webp" alt="pr2 rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("pr2")
```
