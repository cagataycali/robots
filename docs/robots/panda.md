---
title: panda
description: "Franka Emika Panda (7-DOF + gripper)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Franka Emika Panda (7-DOF + gripper)

{{robot_chips:panda}}

<robot-viewer name="panda"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("panda")
print(robot.robot_action_keys("panda"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

```python title="sketch"
robot = Robot("panda", mode="real", port="172.16.0.2")  # FrankaDriver
```

Aliases: `bimanual_panda_gripper`, `bimanual_panda_hand`, `franka`, `franka_emika_panda`, `franka_panda`, `libero_panda`, `oxe_droid`, `oxe_droid_rel`, `oxe_droid_relative_eef_relative_joint`, `single_panda_gripper`.

Gripper actuator `actuator8`: closed at the low end of travel, open at the high end.

## Hardware

**`FrankaDriver`** (the default for this robot) speaks Franka Control Interface (FCI) through `panda-py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#frankadriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_emika_panda), scene `scene.xml`.
