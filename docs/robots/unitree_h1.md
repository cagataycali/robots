---
title: unitree_h1
description: "Unitree H1 Humanoid (19-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree H1 Humanoid (19-DOF)

{{robot_chips:unitree_h1}}

<robot-viewer name="unitree_h1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_h1")
print(robot.robot_action_keys("unitree_h1"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

```python title="sketch"
robot = Robot("unitree_h1", mode="real", port="192.168.123.161", network_interface="eth0")  # Go2Driver
```

Aliases: `h1`.

## Hardware

**`Go2Driver`** (the default for this robot) speaks CycloneDDS through `unitree_sdk2py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#go2driver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/unitree_h1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_h1), scene `scene.xml`.
