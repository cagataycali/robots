---
title: anymal_d
description: "anymal_d (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# anymal_d (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">14 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [ANYbotics/anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description/tree/6adc14720aab583613975e5a9d6d4fa3cfcdd081), compiled for MuJoCo by `strands_robots` on first use: 14 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/anymal_d.webp" alt="anymal_d rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("anymal_d")
```
