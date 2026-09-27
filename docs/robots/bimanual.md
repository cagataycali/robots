---
description: Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.
---

# Bimanual rigs

Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.

```python
from strands_robots import Robot
sim = Robot("aloha")            # Trossen Aloha bimanual - sim only, lerobot ships no ALOHA robot
# Trossen WX-AI - its asset is never auto-downloaded: place
# trossen_wxai/trossen_ai_bimanual.xml under STRANDS_ASSETS_DIR first.
sim = Robot("trossen_wxai")

# Hardware only, no sim twin - one lerobot follower config per arm.
arms = Robot("bi_so_follower", mode="real",      # two SO-101 followers on one Feetech bus
             left_arm_config=..., right_arm_config=...)
arms = Robot("bi_openarm", mode="real",          # two OpenArm followers on CAN
             left_arm_config=..., right_arm_config=...)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:bimanual}}

## See also

- [Arms](arms.md) - single-arm manipulators.
- [Hands](hands.md) - dexterous end-effectors to mount on each arm.
- [Multi-robot mesh](../reference/mesh.md) - pair two single arms via the mesh as an alternative to a single bimanual rig.
