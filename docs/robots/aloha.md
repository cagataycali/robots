---
title: aloha
description: "ALOHA Bimanual (2x ViperX 300s, 14-DOF + 2 grippers)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# ALOHA Bimanual (2x ViperX 300s, 14-DOF + 2 grippers)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip">28 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

<robot-viewer name="aloha"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("aloha")
```

Aliases: `agibot_dual_arm`, `agibot_dual_arm_dexhand`, `agibot_dual_arm_full`, `agibot_dual_arm_gripper`, `agibot_genie1`, `galaxea_r1_pro`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/aloha](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/aloha), scene `scene.xml`.
