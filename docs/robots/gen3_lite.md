---
title: gen3_lite
description: "gen3_lite (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# gen3_lite (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">10 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [Kinovarobotics/ros2_kortex@8bf2034](https://github.com/Kinovarobotics/ros2_kortex/tree/8bf203423911446de28a2248ec87380b7eea2f90), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 10 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/gen3_lite.webp" alt="gen3_lite, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("gen3_lite")
```
