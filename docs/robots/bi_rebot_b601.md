---
title: bi_rebot_b601
description: "Bi-manual reBot B601-DM (dual 6-DOF + gripper, Damiao CAN motors)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Bi-manual reBot B601-DM (dual 6-DOF + gripper, Damiao CAN motors)

<p class="sr-chips"><span class="sr-chip sr-chip-family" data-family="bimanual">Bimanual</span><span class="sr-chip sr-chip-real">real</span><span class="sr-chip sr-chip-driver">driver: lerobot</span></p>

The registry ships no simulation asset for it, so `Robot("bi_rebot_b601")` in the default sim mode refuses by name.

```python title="sketch"
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig

robot = Robot("bi_rebot_b601", mode="real",  # lerobot bi_rebot_b601_follower: one config per arm, no single port
              left_arm_config=RebotB601FollowerConfig(port="/dev/ttyACM0"),
              right_arm_config=RebotB601FollowerConfig(port="/dev/ttyACM1"))
# this lerobot type is on lerobot main, not on PyPI: install lerobot from source
```

Aliases: `bi_rebot_b601_follower`, `dual_rebot_b601`.

## Hardware

**lerobot.** `Robot("bi_rebot_b601", mode="real")` builds lerobot's `bi_rebot_b601_follower` with `pip install 'strands-robots[lerobot]'`; there is no single `port=`; pass `left_arm_config=` and `right_arm_config=`, one `RebotB601FollowerConfig` per arm with its own `port` and `cameras`. The default when `driver=` is not given. Install lerobot from source: the type is not in the PyPI release.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.
