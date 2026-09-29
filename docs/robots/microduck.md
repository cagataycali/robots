---
title: microduck
description: "Pollen Microduck (14-DOF open-source biped, Dynamixel XL330)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Pollen Microduck (14-DOF open-source biped, Dynamixel XL330)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip">15 joints</span><span class="sr-chip sr-chip-sim">sim</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: strands</span></p>

<robot-viewer name="microduck"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("microduck")
```

```python title="sketch"
robot = Robot("microduck", mode="real", port="ssh://radxa@microduck.local")  # MicroduckDriver
```

Aliases: `micro_duck`, `pollen_microduck`.

## Hardware

**`MicroduckDriver`** (the default for this robot) speaks `robotd` JSON-RPC over a unix socket: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#microduckdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Providers written for this body: `microduck`; the rest are in the [policy matrix](../learn/policies/index.md).

Model: [pollen-robotics/microduck_rl/src/mjlab_microduck/robot/microduck](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/robot/microduck), scene `scene.xml`.
