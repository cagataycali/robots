---
title: reachy2
description: "Pollen Reachy 2"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Pollen Reachy 2

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="humanoid">Humanoids</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot</span></p>

The registry ships no simulation asset for it, so `Robot("reachy2")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("reachy2", mode="real", port="/dev/ttyACM0")  # lerobot reachy2
```

## Hardware

**lerobot.** `Robot("reachy2", mode="real")` builds lerobot's `reachy2` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies that ran on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
