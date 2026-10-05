---
title: cassie
description: "Agility Cassie Bipedal Robot"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Agility Cassie Bipedal Robot

{{robot_chips:cassie}}

<robot-viewer name="cassie"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("cassie")
print(robot.robot_action_keys("cassie"))
robot.cleanup()
```

Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).

Aliases: `agility_cassie`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/agility_cassie](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/agility_cassie), scene `scene.xml`.
