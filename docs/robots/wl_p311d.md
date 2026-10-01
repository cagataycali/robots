---
title: wl_p311d
description: "wl_p311d (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# wl_p311d (quadruped URDF from robot_descriptions)

{{robot_chips:wl_p311d}}

URDF from [limxdynamics/robot-description@a097533](https://github.com/limxdynamics/robot-description/tree/a097533372a08298d45af391cbdfc2fd2dc3da6f), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 16 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/wl_p311d.webp" alt="wl_p311d, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("wl_p311d")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("wl_p311d"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
