---
title: earthrover
description: "EarthRover Mini Plus (mobile outdoor navigation)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# EarthRover Mini Plus (mobile outdoor navigation)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile">Mobile bases</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot, strands</span></p>

The registry ships no simulation asset for it, so `Robot("earthrover")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("earthrover", mode="real", port="/dev/ttyACM0")  # lerobot earthrover_mini_plus
robot = Robot("earthrover", mode="real", driver="strands", port="http://localhost:8000")  # EarthRoverDriver
```

Aliases: `earth_rover`, `earthrover_mini_plus`, `frodobots`.

## Hardware

**lerobot.** `Robot("earthrover", mode="real")` builds lerobot's `earthrover_mini_plus` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

**`EarthRoverDriver`** (selected with `driver="strands"`) speaks HTTP to the vendor `earth-rovers-sdk`: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#earthroverdriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
