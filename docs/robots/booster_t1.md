---
title: booster_t1
description: "Booster T1 Humanoid (24-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Booster T1 Humanoid (24-DOF)

{{robot_chips:booster_t1}}

<robot-viewer name="booster_t1"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("booster_t1")
```

```python title="sketch"
robot = Robot("booster_t1", mode="real", port="192.168.10.102")  # BoosterDriver
```

## Hardware

**`BoosterDriver`** (the default for this robot) speaks Booster SDK (`booster_robotics_sdk_python`, DDS): [port, SDK, kwargs and checks](../learn/hardware/drivers.md#boosterdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/booster_t1](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/booster_t1), scene `scene.xml`.
