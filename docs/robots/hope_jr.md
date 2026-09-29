---
title: hope_jr
description: "HopeJR Arm (high-DOF anthropomorphic arm, Feetech)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# HopeJR Arm (high-DOF anthropomorphic arm, Feetech)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot, strands</span></p>

The registry ships no simulation asset for it, so `Robot("hope_jr")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("hope_jr", mode="real", port="/dev/ttyACM0")  # lerobot hope_jr_arm
robot = Robot("hope_jr", mode="real", driver="strands", port="/dev/ttyACM0")  # FeetechDriver
```

## Hardware

**lerobot.** `Robot("hope_jr", mode="real")` builds lerobot's `hope_jr_arm` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

**`FeetechDriver`** (selected with `driver="strands"`) speaks Feetech STS/SMS serial bus: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#feetechdriver).
