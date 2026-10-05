---
title: open_duck_mini
description: "Open Duck Mini V2 (16-DOF expressive biped, Feetech servos)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Open Duck Mini V2 (16-DOF expressive biped, Feetech servos)

{{robot_chips:open_duck_mini}}

<robot-viewer name="open_duck_mini"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("open_duck_mini", position=[0.0, 0.0, 0.0262])
print(robot.robot_joint_names("open_duck_mini"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

```python title="sketch"
robot = Robot("open_duck_mini", mode="real", port="/dev/ttyACM0")  # FeetechDriver
```

Aliases: `bdx`, `mini_bdx`, `open_duck`, `open_duck_mini_v2`, `open_duck_v2`.

## Hardware

**`FeetechDriver`** (the default for this robot) speaks Feetech STS/SMS serial bus: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#feetechdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [apirrone/Open_Duck_Mini/mini_bdx/robots/open_duck_mini_v2](https://github.com/apirrone/Open_Duck_Mini/tree/b23317a485b3cec7d8417f352478778b3475173c/mini_bdx/robots/open_duck_mini_v2), scene `scene.xml`.
