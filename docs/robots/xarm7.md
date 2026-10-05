---
title: xarm7
description: "UFactory xArm 7 (7-DOF + gripper)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# UFactory xArm 7 (7-DOF + gripper)

{{robot_chips:xarm7}}

<robot-viewer name="xarm7"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("xarm7")
print(robot.robot_joint_names("xarm7"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run [a checkpoint](../start/first-policy.md) on it.

```python title="sketch"
robot = Robot("xarm7", mode="real", port="192.168.1.185")  # XArmDriver
```

Aliases: `ufactory_xarm7`.

## Hardware

**`XArmDriver`** (the default for this robot) speaks TCP through `xarm-python-sdk`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#xarmdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/ufactory_xarm7](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/ufactory_xarm7), scene `scene.xml`.
