---
title: unitree_go2
description: "Unitree Go2 Quadruped"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Unitree Go2 Quadruped

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip">40 joints</span><span class="sr-chip sr-chip-sim">sim</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: strands</span></p>

<robot-viewer name="unitree_go2"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("unitree_go2")
```

```python title="sketch"
robot = Robot("unitree_go2", mode="real", port="192.168.123.161", network_interface="eth0")  # Go2Driver
```

Aliases: `go2`.

## Hardware

**`Go2Driver`** (the default for this robot) speaks CycloneDDS through `unitree_sdk2py`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#go2driver).

Model: [google-deepmind/mujoco_menagerie/unitree_go2](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/unitree_go2), scene `scene.xml`.
