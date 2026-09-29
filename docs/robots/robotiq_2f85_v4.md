---
title: robotiq_2f85_v4
description: "Robotiq 2F-85 v4 Gripper (updated model)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Robotiq 2F-85 v4 Gripper (updated model)

{{robot_chips:robotiq_2f85_v4}}

<robot-viewer name="robotiq_2f85_v4"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("robotiq_2f85_v4")
```

```python title="sketch"
robot = Robot("robotiq_2f85_v4", mode="real", port="192.168.1.11")  # RobotiqDriver
```

## Hardware

**`RobotiqDriver`** (the default for this robot) speaks Modbus TCP: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#robotiqdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/robotiq_2f85_v4](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/robotiq_2f85_v4), scene `scene.xml`.
