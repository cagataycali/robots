---
title: ur5e
description: "Universal Robots UR5e (6-DOF industrial)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Universal Robots UR5e (6-DOF industrial)

{{robot_chips:ur5e}}

<robot-viewer name="ur5e"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("ur5e")
```

```python title="sketch"
robot = Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")  # URDriver
```

## Hardware

**`URDriver`** (selected with `driver="strands"`) speaks RTDE through `ur_rtde`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#urdriver).

Model: [google-deepmind/mujoco_menagerie/universal_robots_ur5e](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/universal_robots_ur5e), scene `scene.xml`.
