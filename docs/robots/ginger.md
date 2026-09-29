---
title: ginger
description: "ginger (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ginger (mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">49 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [Rayckey/GingerURDF@6a1307c](https://github.com/Rayckey/GingerURDF/tree/6a1307cd0ee2b77c82f8839cdce3a2e2eed2bd8f), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 49 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/ginger.webp" alt="ginger, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("ginger")
```
