---
title: yahboom_m3pro
description: "Yahboom ROSMASTER M3 Pro (mecanum base + DOFBOT-Pro 6-DOF arm: 5 servos + gripper, 9 actuators)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Yahboom ROSMASTER M3 Pro (mecanum base + DOFBOT-Pro 6-DOF arm: 5 servos + gripper, 9 actuators)

{{robot_chips:yahboom_m3pro}}

<robot-viewer name="yahboom_m3pro"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("yahboom_m3pro")
```

```python title="sketch"
robot = Robot("yahboom_m3pro", mode="real", port="192.168.1.50:9090")  # YahboomM3ProDriver
```

Aliases: `m3pro`, `m3_pro`, `rosmaster_m3_pro`, `rosmaster_m3pro`, `yahboom_m3_pro`, `yahboom_rosmaster_m3_pro`.

Gripper actuator `gripper`: closed at the low end of travel, open at the high end.

## Hardware

**`YahboomM3ProDriver`** (the default for this robot) speaks the robot's ROS 2 graph, over rosbridge or in-process `rclpy`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#yahboomm3prodriver).

Model: [dimwael/yahboom_m3pro_description/mjcf](https://github.com/dimwael/yahboom_m3pro_description/tree/bdac682e57a8eeaf7b18eece44bfbfc54de98e8a/mjcf), scene `scene.xml`.
