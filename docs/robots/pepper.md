---
title: pepper
description: "pepper (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# pepper (mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">45 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [jrl-umi3218/pepper_description@cd9715b](https://github.com/jrl-umi3218/pepper_description/tree/cd9715bb5df7ad57445d953db7b1924255305944), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 45 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/pepper.webp" alt="pepper, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("pepper")
```
