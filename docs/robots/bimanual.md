---
description: Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.
---

# Bimanual rigs

Two-arm robots - Aloha, bimanual SO-ARM, Trossen WX-AI, OpenArm bimanual.

```python
from strands_robots import Robot
sim = Robot("aloha")            # Trossen Aloha bimanual
sim = Robot("bi_openarm")       # OpenArm bimanual
sim = Robot("trossen_wxai")     # Trossen WX-AI

# Two SO-101 followers on one Feetech bus - hardware only, no sim twin.
arms = Robot("bi_so_follower", mode="real",
             left_arm_config=..., right_arm_config=...)
```

## Catalog

| Name | Description | Joints | Aliases |
|------|-------------|-------:|---------|
| `aloha` | ALOHA Bimanual (2x ViperX 300s, 14-DOF + 2 grippers) _(simulation-only: lerobot ships no ALOHA robot)_ | 28 | `agibot_dual_arm`, `agibot_dual_arm_dexhand`, `agibot_dual_arm_full`, `agibot_dual_arm_gripper`, `agibot_genie1`, `galaxea_r1_pro` |
| `bi_openarm` | Bi-manual OpenArm (dual-arm coordination) _(hardware-only, no sim asset)_ | ? | `bi_openarm_follower`, `dual_openarm`, `openarm_bimanual` |
| `bi_rebot_b601` | Bi-manual reBot B601-DM (dual 6-DOF + gripper, Damiao CAN motors) _(hardware-only, no sim asset)_ | ? | `bi_rebot_b601_follower`, `dual_rebot_b601` |
| `bi_so_follower` | Bimanual SO-ARM follower (2x SO-100/SO-101, 6-DOF each, Feetech STS3215) _(hardware-only, no sim asset)_ | ? | `bi_so100`, `bi_so101` |
| `trossen_wxai` | Trossen WidowX AI Bimanual | 17 | `trossen_ai_bimanual` |

## Featured renders

### `aloha`

![aloha](../assets/sim_render_aloha.png){ width=400 }

_ALOHA Bimanual (2x ViperX 300s, 14-DOF + 2 grippers)_

### `trossen_wxai`

![trossen_wxai](../assets/sim_render_trossen_wxai.png){ width=400 }

_Trossen WidowX AI Bimanual_

## See also

- [Arms](arms.md) - single-arm manipulators.
- [Hands](hands.md) - dexterous end-effectors to mount on each arm.
- [Multi-robot mesh](../mesh.md) - pair two single arms via the mesh as an alternative to a single bimanual rig.
