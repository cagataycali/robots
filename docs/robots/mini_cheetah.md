---
title: mini_cheetah
description: "mini_cheetah (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# mini_cheetah (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">12 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

A URDF from [Derek-TH-Wang/mini_cheetah_urdf](https://github.com/Derek-TH-Wang/mini_cheetah_urdf/tree/1988bceb26e81f28594a16e7d5e6abe5cbb27ace), compiled for MuJoCo by `strands_robots` on first use: 12 position actuators sized from the URDF effort limits, a free-floating base, a floor and a light. See [URDF robots](../learn/simulation/urdf.md).

<img class="sr-thumb" src="../assets/img/robots/mini_cheetah.webp" alt="mini_cheetah rendered in MuJoCo" width="400">

The MJCF is compiled on your machine, so this page has no 3D view; the thumbnail is a local render.

```python
from strands_robots import Robot

robot = Robot("mini_cheetah")
```
