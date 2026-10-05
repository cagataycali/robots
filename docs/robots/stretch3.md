---
title: stretch3
description: "Hello Robot Stretch 3 (mobile manipulator)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Hello Robot Stretch 3 (mobile manipulator)

{{robot_chips:stretch3}}

<robot-viewer name="stretch3"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("stretch3")
print(robot.robot_action_keys("stretch3"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

```python title="sketch"
robot = Robot("stretch3", mode="real")  # StretchDriver
```

Aliases: `hello_robot_stretch`, `hello_robot_stretch_3`.

## Hardware

**`StretchDriver`** (the default for this robot) speaks USB through `stretch_body`, on the robot's own computer: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#stretchdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/hello_robot_stretch_3](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/hello_robot_stretch_3), scene `scene.xml`.
