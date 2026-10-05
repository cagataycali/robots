---
title: spot
description: "Boston Dynamics Spot (with arm)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Boston Dynamics Spot (with arm)

{{robot_chips:spot}}

<robot-viewer name="spot"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("spot")
print(robot.robot_action_keys("spot"))
robot.cleanup()
```

The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.

```python title="sketch"
robot = Robot("spot", mode="real", port="192.168.80.3")  # SpotDriver
```

Aliases: `boston_dynamics_spot`.

## Hardware

**`SpotDriver`** (the default for this robot) speaks gRPC through `bosdyn-client`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#spotdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/boston_dynamics_spot](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/boston_dynamics_spot), scene `scene_arm.xml`.
