---
title: jaxon
description: "jaxon (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# jaxon (humanoid URDF from robot_descriptions)

{{robot_chips:jaxon}}

URDF from [robot-descriptions/jaxon_description@4a0cb7a](https://github.com/robot-descriptions/jaxon_description/tree/4a0cb7a4a737864312f8d6e3f89823a741539bfc), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 38 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/jaxon.webp" alt="jaxon, a local MuJoCo render" loading="lazy" width="640" height="480">

```python
from strands_robots import Robot

robot = Robot("jaxon")
```

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
