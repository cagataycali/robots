---
title: mini_cheetah
description: "mini_cheetah (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# mini_cheetah (quadruped URDF from robot_descriptions)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">12 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

URDF from [Derek-TH-Wang/mini_cheetah_urdf@1988bce](https://github.com/Derek-TH-Wang/mini_cheetah_urdf/tree/1988bceb26e81f28594a16e7d5e6abe5cbb27ace), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 12 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/mini_cheetah.webp" alt="mini_cheetah, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("mini_cheetah")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
