---
title: spryped
description: "spryped (biped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# spryped (biped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">8 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [bbokser/spryped@f360a6b](https://github.com/bbokser/spryped/tree/f360a6b78667a4d97c86cad465ef8f4c9512462b), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 8 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/spryped.webp" alt="spryped, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("spryped")
```
