---
title: kuka_iiwa
description: "KUKA LBR iiwa 14 (7-DOF collaborative)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# KUKA LBR iiwa 14 (7-DOF collaborative)

{{robot_chips:kuka_iiwa}}

<robot-viewer name="kuka_iiwa"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("kuka_iiwa")
print(robot.robot_action_keys("kuka_iiwa"))
robot.cleanup()
```

Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).

Aliases: `iiwa`, `iiwa14`, `kuka_iiwa_14`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/kuka_iiwa_14), scene `scene.xml`.
