---
title: barrett_hand
description: "barrett_hand (end_effector URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# barrett_hand (end_effector URDF from robot_descriptions)

{{robot_chips:barrett_hand}}

URDF from [jhu-lcsr-attic/bhand_model@937f418](https://github.com/jhu-lcsr-attic/bhand_model/tree/937f4186d6458bd682a7dae825fb6f4efe56ec69), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 8 position actuators, fixed base.

<img class="sr-thumb" src="../assets/img/robots/barrett_hand.webp" alt="barrett_hand, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("barrett_hand")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("barrett_hand"))
robot.cleanup()
```

`robot_action_keys` lists the actuators a policy drives; command them by name with `send_action` or from a [policy](../learn/policies/index.md).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
