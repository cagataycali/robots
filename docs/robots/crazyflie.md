---
title: crazyflie
description: "Bitcraze Crazyflie 2 Nano-Quadcopter"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Bitcraze Crazyflie 2 Nano-Quadcopter

{{robot_chips:crazyflie}}

<robot-viewer name="crazyflie"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("crazyflie")
```

```python title="sketch"
robot = Robot("crazyflie", mode="real", port="radio://0/80/2M/E7E7E7E7E7")  # CrazyflieDriver
```

Aliases: `cf2`, `bitcraze_crazyflie`.

## Hardware

**`CrazyflieDriver`** (the default for this robot) speaks CRTP over a Crazyradio through `cflib`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#crazyfliedriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/bitcraze_crazyflie_2](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/bitcraze_crazyflie_2), scene `scene.xml`.
