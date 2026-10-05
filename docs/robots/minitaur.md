---
title: minitaur
description: "minitaur (quadruped URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# minitaur (quadruped URDF from robot_descriptions)

{{robot_chips:minitaur}}

URDF from [bulletphysics/bullet3@7dee343](https://github.com/bulletphysics/bullet3/tree/7dee3436e747958e7088dfdcea0e4ae031ce619e), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 16 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/minitaur.webp" alt="minitaur, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("minitaur")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("minitaur"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
