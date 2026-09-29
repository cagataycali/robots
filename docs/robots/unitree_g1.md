---
title: unitree_g1
description: "Unitree G1 Humanoid (29-DOF + dexterous hands)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree G1 Humanoid (29-DOF + dexterous hands)

{{robot_chips:unitree_g1}}

<robot-viewer name="unitree_g1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_g1")
```

```python title="sketch"
robot = Robot("unitree_g1", mode="real", driver="lerobot", port="/dev/ttyACM0")  # lerobot unitree_g1
robot = Robot("unitree_g1", mode="real", port="192.168.123.164", network_interface="eth0")  # G1Driver
```

Aliases: `g1`, `g1_wbc`, `real_g1_relative_eef_relative_joints`, `unitree_g1_full_body`, `unitree_g1_locomanip`, `unitree_g1_real`, `unitree_g1_sonic`, `unitree_g1_wbc`.

## Hardware

**lerobot.** `Robot("unitree_g1", mode="real")` builds lerobot's `unitree_g1` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict.

**`G1Driver`** (the default for this robot) speaks CycloneDDS through `unitree_sdk2py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#g1driver).

## Policies

Providers written for this body: `wbc`, `wbc_gait`, `kimodo`, `protomotions`; the rest are in the [policy matrix](../learn/policies/index.md).

Model: [google-deepmind/mujoco_menagerie/unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_g1), scene `scene.xml`.
