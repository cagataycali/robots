---
title: microduck
description: "Pollen Microduck (14-DOF open-source biped, Dynamixel XL330)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Pollen Microduck (14-DOF open-source biped, Dynamixel XL330)

{{robot_chips:microduck}}

<robot-viewer name="microduck"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("microduck")
print(robot.robot_action_keys("microduck"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

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
