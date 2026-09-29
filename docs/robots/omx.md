---
title: omx
description: "OMX Robot Arm (ROBOTIS, CAN bus motors)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# OMX Robot Arm (ROBOTIS, CAN bus motors)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="arm">Arms</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot</span></p>

The registry ships no simulation asset for it, so `Robot("omx")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("omx", mode="real", port="/dev/ttyACM0")  # lerobot omx_follower
```

Aliases: `omx_follower`, `omx_robot`, `robotis_omx`.

## Hardware

**lerobot.** `Robot("omx", mode="real")` builds lerobot's `omx_follower` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
