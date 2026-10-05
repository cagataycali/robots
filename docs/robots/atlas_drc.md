---
title: atlas_drc
description: "atlas_drc (humanoid URDF from robot_descriptions)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# atlas_drc (humanoid URDF from robot_descriptions)

{{robot_chips:atlas_drc}}

URDF from [RobotLocomotion/drake@7abea05](https://github.com/RobotLocomotion/drake/tree/7abea0556ede980a5077fe1a8cfbae59b57c7c27), [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: 30 position actuators, floating base.

<img class="sr-thumb" src="../assets/img/robots/atlas_drc.webp" alt="atlas_drc, a local MuJoCo render" loading="lazy" width="640" height="480">

```python title="sketch"
from strands_robots import Robot

robot = Robot("atlas_drc")  # clones the description and compiles the URDF on first use
print(robot.robot_action_keys("atlas_drc"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
