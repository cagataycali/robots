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
print(robot.robot_joint_names("spot"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

Aliases: `boston_dynamics_spot`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/boston_dynamics_spot](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/boston_dynamics_spot), scene `scene_arm.xml`.
