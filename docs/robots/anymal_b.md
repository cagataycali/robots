---
title: anymal_b
description: "ANYbotics ANYmal B Quadruped (12-DOF)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ANYbotics ANYmal B Quadruped (12-DOF)

{{robot_chips:anymal_b}}

<robot-viewer name="anymal_b"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("anymal_b")
print(robot.robot_action_keys("anymal_b"))
robot.cleanup()
```

Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).

Aliases: `anybotics_anymal_b`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/anybotics_anymal_b](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/anybotics_anymal_b), scene `scene.xml`.
