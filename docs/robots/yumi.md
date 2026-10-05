---
title: yumi
description: "yumi (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# yumi (dual_arm URDF from robot_descriptions)

{{robot_chips:yumi}}

URDF from [ankurhanda/robot-assets@12f1a3c](https://github.com/ankurhanda/robot-assets/tree/12f1a3c89c9975194551afaed0dfae1e09fdb27c), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 18 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/yumi.webp" alt="yumi, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("yumi")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("yumi"))
robot.cleanup()
```

One action dict drives both arms; [composition](../learn/policies/index.md) runs a policy per arm.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
