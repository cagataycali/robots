---
title: hope_jr_hand
description: "HopeJR Hand (dexterous anthropomorphic hand, Feetech)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# HopeJR Hand (dexterous anthropomorphic hand, Feetech)

{{robot_chips:hope_jr_hand}}

The registry ships no simulation asset for it, so `Robot("hope_jr_hand")` in the default sim mode refuses by name.

```python title="sketch"
robot = Robot("hope_jr_hand", mode="real", port="/dev/ttyACM0")  # lerobot hope_jr_hand
```

Aliases: `hopejr_hand`, `hope_junior_hand`.

## Hardware

**lerobot.** `Robot("hope_jr_hand", mode="real")` builds lerobot's `hope_jr_hand` with `pip install 'strands-robots[lerobot]'`; `port=` is the serial device, `cameras=` the lerobot camera dict. The default when `driver=` is not given.
