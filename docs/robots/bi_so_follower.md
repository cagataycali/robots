---
title: bi_so_follower
description: "Bimanual SO-ARM follower (2x SO-100/SO-101, 6-DOF each, Feetech STS3215)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Bimanual SO-ARM follower (2x SO-100/SO-101, 6-DOF each, Feetech STS3215)

{{robot_chips:bi_so_follower}}

The registry ships no simulation asset for it, so `Robot("bi_so_follower")` in the default sim mode refuses by name.

```python title="sketch"
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

robot = Robot("bi_so_follower", mode="real",  # lerobot bi_so_follower: one config per arm, no single port
              left_arm_config=SOFollowerConfig(port="/dev/ttyACM0"),
              right_arm_config=SOFollowerConfig(port="/dev/ttyACM1"))
```

Aliases: `bi_so100`, `bi_so101`.

## Hardware

**lerobot.** `Robot("bi_so_follower", mode="real")` builds lerobot's `bi_so_follower` with `pip install 'strands-robots[lerobot]'`; there is no single `port=`; pass `left_arm_config=` and `right_arm_config=`, one `SOFollowerConfig` per arm with its own `port` and `cameras`. The default when `driver=` is not given.
