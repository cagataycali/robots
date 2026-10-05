---
title: xarm6
description: "xarm6 (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# xarm6 (arm URDF from robot_descriptions)

{{robot_chips:xarm6}}

URDF from [xArm-Developer/xarm_ros2@5bb832f](https://github.com/xArm-Developer/xarm_ros2/tree/5bb832f72ca665f1236a9d8ed1c3a82f308db489), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/xarm6.webp" alt="xarm6, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("xarm6")  # clones the description and compiles the URDF on first use
print(robot.robot_joint_names("xarm6"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
