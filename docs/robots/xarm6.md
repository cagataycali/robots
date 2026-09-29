---
title: xarm6
description: "xarm6 (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# xarm6 (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2/tree/5bb832f72ca665f1236a9d8ed1c3a82f308db489), compiled for MuJoCo by `strands_robots` on first use: 6 position actuators sized from the URDF effort limits, a fixed base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/xarm6.webp" alt="xarm6 rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("xarm6")
```
