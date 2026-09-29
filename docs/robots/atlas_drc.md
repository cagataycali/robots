---
title: atlas_drc
description: "atlas_drc (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# atlas_drc (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">30 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [RobotLocomotion/drake@7abea05](https://github.com/RobotLocomotion/drake/tree/7abea0556ede980a5077fe1a8cfbae59b57c7c27), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 30 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/atlas_drc.webp" alt="atlas_drc, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("atlas_drc")
```
