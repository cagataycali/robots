---
title: nextage
description: "nextage (dual_arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# nextage (dual_arm URDF from robot_descriptions)

{{robot_chips:nextage}}

URDF from [tork-a/rtmros_nextage@ac270fb](https://github.com/tork-a/rtmros_nextage/tree/ac270fb969fa54abeb6863f9b388a9e20c1f14e0), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 15 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/nextage.webp" alt="nextage, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("nextage")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("nextage"))
robot.cleanup()
```

One action dict drives both arms; [composition](../learn/policies/index.md) runs a policy per arm.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
