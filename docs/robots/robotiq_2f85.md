---
title: robotiq_2f85
description: "Robotiq 2F-85 Gripper (2-finger adaptive)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Robotiq 2F-85 Gripper (2-finger adaptive)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="hand">Hands and grippers</span><span class="sr-chip">16 joints</span><span class="sr-chip sr-chip-sim">sim</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: strands</span></p>

<robot-viewer name="robotiq_2f85"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("robotiq_2f85")
```

```python title="sketch"
robot = Robot("robotiq_2f85", mode="real", port="192.168.1.11")  # RobotiqDriver
```

Aliases: `robotiq`.

## Hardware

**`RobotiqDriver`** (the default for this robot) speaks Modbus TCP: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#robotiqdriver).

Model: [google-deepmind/mujoco_menagerie/robotiq_2f85](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/robotiq_2f85), scene `scene.xml`.
