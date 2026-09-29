---
title: fer
description: "fer (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# fer (arm URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">9 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [frankarobotics/franka_description@1aa4fd3](https://github.com/frankarobotics/franka_description/tree/1aa4fd30e6e274cbf5e986a5af8004df32bad284), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 9 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/fer.webp" alt="fer, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("fer")
```
