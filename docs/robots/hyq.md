---
title: hyq
description: "hyq (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# hyq (quadruped URDF from robot_descriptions)

{{robot_chips:hyq}}

URDF from [Gepetto/example-robot-data@d0d9098](https://github.com/Gepetto/example-robot-data/tree/d0d9098d752014aec3725b07766962acf06c5418), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 12 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/hyq.webp" alt="hyq, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("hyq")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("hyq"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
