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

| Checkpoint | Provider | Where | What happened |
|---|---|---|---|
| alpha_walking.onnx (Pollen's RL gait) | `microduck` | sim | 250 of 250 actions, 0 errors; a 0.15 m/s command moved the base 0.35 m forward in 5 s, upright. Source: examples/microduck/microduck_walk_sim.py --vx 0.15 --duration 5, 2026-10-07 |
| velstand.onnx (Pollen's default walk) | `microduck` | sim | 250 of 250 actions, 0 errors; 0.15 m/s: 0.30 m forward, 0.15 m left in 5 s, upright; zero command: 9 mm drift. Source: examples/microduck/microduck_walk_sim.py --onnx velstand.onnx --vx 0.15 or 0 --duration 5, 2026-10-07 |

Providers written for this body: `microduck`; the rest are in the [policy matrix](../learn/policies/index.md).

Model: [pollen-robotics/microduck_rl/src/mjlab_microduck/robot/microduck](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/robot/microduck), scene `scene.xml`.
