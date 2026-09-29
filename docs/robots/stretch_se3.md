---
title: stretch_se3
description: "stretch_se3 (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# stretch_se3 (mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">14 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [hello-robot/stretch_urdf](https://github.com/hello-robot/stretch_urdf/tree/1b7cbbce808c25465017ce0a53a4173fcf97b11c), compiled for MuJoCo by `strands_robots` on first use: 14 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/stretch_se3.webp" alt="stretch_se3 rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("stretch_se3")
```
