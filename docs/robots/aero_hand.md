---
title: aero_hand
description: "Tetheria Aero Hand Open (16-DOF dexterous)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Tetheria Aero Hand Open (16-DOF dexterous)

{{robot_chips:aero_hand}}

<robot-viewer name="aero_hand"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("aero_hand", position=[0.0, 0.0, 0.1083])
print(robot.robot_action_keys("aero_hand"))
robot.cleanup()
```

`robot_action_keys` lists the actuators a policy drives; command them by name with `send_action` or from a [policy](../learn/policies/index.md).

Aliases: `tetheria_aero_hand`, `aero_hand_open`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/tetheria_aero_hand_open](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/tetheria_aero_hand_open), scene `scene_right.xml`.
