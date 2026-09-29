---
title: rhea
description: "rhea (biped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# rhea (biped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">6 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [G-Levine/rhea_description@1dc0f1a](https://github.com/G-Levine/rhea_description/tree/1dc0f1abcf51b5d8a8f7ff8a548399ff0df1414f), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/rhea.webp" alt="rhea, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("rhea")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
