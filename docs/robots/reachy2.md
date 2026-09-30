---
title: reachy2
description: "Pollen Reachy 2"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Pollen Reachy 2

{{robot_chips:reachy2}}

The registry ships no simulation asset for it, so `Robot("reachy2")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("reachy2", mode="real", ip_address="192.168.1.42")  # lerobot reachy2
```

## Hardware

**lerobot.** `Robot("reachy2", mode="real")` builds lerobot's `reachy2` with `pip install 'strands-robots[lerobot]'`; `ip_address=` is the robot's address (gRPC, `port=` 50065), `cameras=` the lerobot camera dict. The default when `driver=` is not given.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
