---
title: bambot
description: "bambot (mobile_manipulator URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# bambot (mobile_manipulator URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">15 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [timqian/bambot@04d9026](https://github.com/timqian/bambot/tree/04d902653794f9f72eeabb09ec90a9af8e397c5b), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 15 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/bambot.webp" alt="bambot, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("bambot")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
