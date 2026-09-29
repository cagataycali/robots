---
title: fr3
description: "Franka Research 3 (7-DOF + gripper)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Franka Research 3 (7-DOF + gripper)

{{robot_chips:fr3}}

<robot-viewer name="fr3"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("fr3")
```

```python title="sketch"
robot = Robot("fr3", mode="real", driver="strands", port="172.16.0.2")  # FrankaDriver
```

Aliases: `franka_fr3`.

## Hardware

**`FrankaDriver`** (selected with `driver="strands"`) speaks Franka Control Interface (FCI) through `panda-py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#frankadriver).

Model: [google-deepmind/mujoco_menagerie/franka_fr3](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_fr3), scene `scene.xml`.
