---
description: Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.
---

# Bimanual rigs

Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.

```python
from strands_robots import Robot
sim = Robot("aloha")            # Trossen Aloha bimanual - sim only, lerobot ships no ALOHA robot
sim = Robot("bi_openarm")       # OpenArm bimanual
sim = Robot("trossen_wxai")     # Trossen WX-AI

# Two SO-101 followers on one Feetech bus - hardware only, no sim twin.
arms = Robot("bi_so_follower", mode="real",
             left_arm_config=..., right_arm_config=...)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:bimanual}}

## See also

- [Arms](arms.md) - single-arm manipulators.
- [Hands](hands.md) - dexterous end-effectors to mount on each arm.
- [Multi-robot mesh](../mesh.md) - pair two single arms via the mesh as an alternative to a single bimanual rig.
