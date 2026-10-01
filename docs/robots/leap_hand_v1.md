---
title: leap_hand_v1
description: "leap_hand_v1 (end_effector URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# leap_hand_v1 (end_effector URDF from robot_descriptions)

{{robot_chips:leap_hand_v1}}

URDF from [leap-hand/LEAP_Hand_Sim@150bc3d](https://github.com/leap-hand/LEAP_Hand_Sim/tree/150bc3d4b61fd6619193ba5a8ef209f3609ced89), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 16 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/leap_hand_v1.webp" alt="leap_hand_v1, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("leap_hand_v1")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("leap_hand_v1"))
robot.cleanup()
```

`robot_joint_names` lists the finger joints a policy drives; set them by name with `set_joint_positions` or from a [policy](../learn/policies/index.md).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
