---
title: lekiwi_client
description: "LeKiwi networked client (drives a remote LeKiwi host over ZMQ)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# LeKiwi networked client (drives a remote LeKiwi host over ZMQ)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="mobile_manip">Mobile manipulators</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot</span></p>

The registry ships no simulation asset for it, so `Robot("lekiwi_client")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("lekiwi_client", mode="real", port="/dev/ttyACM0")  # lerobot lekiwi_client
```

Aliases: `lekiwi_remote`, `lekiwi_net`.

## Hardware

**lerobot.** `Robot("lekiwi_client", mode="real")` builds lerobot's `lekiwi_client` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies that ran on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
