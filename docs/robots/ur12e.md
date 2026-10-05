---
title: ur12e
description: "ur12e (arm URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ur12e (arm URDF from robot_descriptions)

{{robot_chips:ur12e}}

URDF from [UniversalRobots/Universal_Robots_ROS2_Description@22f055d](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/22f055da2fa7e2158254426107d1f257fd56aebb), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 6 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/ur12e.webp" alt="ur12e, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("ur12e")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("ur12e"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

```python title="sketch"
robot = Robot("ur12e", mode="real", port="192.168.1.10")  # URDriver
```

## Hardware

**`URDriver`** (the default for this robot) speaks RTDE through `ur_rtde`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#urdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
