---
title: rebot_b601
description: "Seeed Studio reBot B601-DM (6-DOF + gripper, Damiao CAN motors)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Seeed Studio reBot B601-DM (6-DOF + gripper, Damiao CAN motors)

{{robot_chips:rebot_b601}}

The registry ships no simulation asset for it, so `Robot("rebot_b601")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("rebot_b601", mode="real", port="/dev/ttyACM0")  # lerobot rebot_b601_follower
# this lerobot type is on lerobot main, not on PyPI: install lerobot from source
```

Aliases: `rebot_b601_follower`, `seeed_rebot_b601`, `b601_dm`.

## Hardware

**lerobot.** `Robot("rebot_b601", mode="real")` builds lerobot's `rebot_b601_follower` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given. Install lerobot from source: the type is not in the PyPI release.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
