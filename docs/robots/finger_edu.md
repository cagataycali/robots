---
title: finger_edu
description: "finger_edu (educational URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# finger_edu (educational URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip">3 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [Gepetto/example-robot-data@d0d9098](https://github.com/Gepetto/example-robot-data/tree/d0d9098d752014aec3725b07766962acf06c5418), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 3 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/finger_edu.webp" alt="finger_edu, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("finger_edu")
```
