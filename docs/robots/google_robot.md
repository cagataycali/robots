---
title: google_robot
description: "Google Robot (mobile base + arm, RT-X)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Google Robot (mobile base + arm, RT-X)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip">10 joints</span><span class="sr-chip sr-chip-sim">sim only</span></p>

<robot-viewer name="google_robot"></robot-viewer>

The model is not fetched for you (`auto_download: false`): place `google_robot/robot.xml` under `~/.strands_robots/assets/` (or `$STRANDS_ASSETS_DIR`) first, or the call refuses with "model file is not on disk".

```python title="sketch"
from strands_robots import Robot

robot = Robot("google_robot")  # needs ~/.strands_robots/assets/google_robot/robot.xml on disk
```

Aliases: `oxe_google`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/google_robot](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/google_robot), scene `scene.xml`.
