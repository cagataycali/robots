---
title: bambot
description: "bambot (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# bambot (mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">15 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [timqian/bambot](https://github.com/timqian/bambot/tree/04d902653794f9f72eeabb09ec90a9af8e397c5b), compiled for MuJoCo by `strands_robots` on first use: 15 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/bambot.webp" alt="bambot rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("bambot")
```
