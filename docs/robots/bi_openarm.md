---
title: bi_openarm
description: "Bi-manual OpenArm (dual-arm coordination)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Bi-manual OpenArm (dual-arm coordination)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot</span></p>

The registry ships no simulation asset for it, so `Robot("bi_openarm")` in the default sim mode refuses by name.

```python title="sketch"
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfig

robot = Robot("bi_openarm", mode="real",  # lerobot bi_openarm_follower: one config per arm, no single port
              left_arm_config=OpenArmFollowerConfig(port="/dev/ttyACM0"),
              right_arm_config=OpenArmFollowerConfig(port="/dev/ttyACM1"))
```

Aliases: `bi_openarm_follower`, `dual_openarm`, `openarm_bimanual`.

## Hardware

**lerobot.** `Robot("bi_openarm", mode="real")` builds lerobot's `bi_openarm_follower` with `pip install 'strands-robots[lerobot]'`; there is no single `port=`; pass `left_arm_config=` and `right_arm_config=`, one `OpenArmFollowerConfig` per arm with its own `port` and `cameras`. The default when `driver=` is not given.

## Policies that ran on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
