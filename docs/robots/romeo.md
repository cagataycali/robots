---
title: romeo
description: "romeo (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# romeo (humanoid URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">61 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [ros-aldebaran/romeo_robot@0.1.5](https://github.com/ros-aldebaran/romeo_robot/tree/0.1.5), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 61 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/romeo.webp" alt="romeo, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("romeo")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
