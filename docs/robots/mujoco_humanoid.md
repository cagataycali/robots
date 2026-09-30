---
title: mujoco_humanoid
description: "MuJoCo Humanoid (21-DOF reference model)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# MuJoCo Humanoid (21-DOF reference model)

{{robot_chips:mujoco_humanoid}}

<robot-viewer name="mujoco_humanoid"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("mujoco_humanoid")
print(robot.robot_joint_names("mujoco_humanoid"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `humanoid`, `mjc_humanoid`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco/model/humanoid](https://github.com/google-deepmind/mujoco/tree/ad0dc0de5e10a075a2c65be629e9a8d557d383a6/model/humanoid), scene `humanoid.xml`.
