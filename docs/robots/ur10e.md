---
title: ur10e
description: "Universal Robots UR10e (6-DOF industrial)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Universal Robots UR10e (6-DOF industrial)

{{robot_chips:ur10e}}

<robot-viewer name="ur10e"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("ur10e", position=[0.0, 0.0, 0.027])
print(robot.robot_action_keys("ur10e"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

```python title="sketch"
robot = Robot("ur10e", mode="real", port="192.168.1.10")  # URDriver
```

## Hardware

**`URDriver`** (the default for this robot) speaks RTDE through `ur_rtde`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#urdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/universal_robots_ur10e](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/universal_robots_ur10e), scene `scene.xml`.
