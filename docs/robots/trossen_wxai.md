---
title: trossen_wxai
description: "Trossen WidowX AI Bimanual"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Trossen WidowX AI Bimanual

{{robot_chips:trossen_wxai}}

<robot-viewer name="trossen_wxai"></robot-viewer>

The model is not fetched for you (`auto_download: false`): place `trossen_wxai/trossen_ai_bimanual.xml` under `~/.strands_robots/assets/` (or `$STRANDS_ASSETS_DIR`) first, or the call refuses with "model file is not on disk".

```python title="sketch"
from strands_robots import Robot

robot = Robot("trossen_wxai")  # needs ~/.strands_robots/assets/trossen_wxai/trossen_ai_bimanual.xml on disk
```

One action dict drives both arms; [composition](../learn/policies/index.md) runs a policy per arm.

Aliases: `trossen_ai_bimanual`.

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [google-deepmind/mujoco_menagerie/trossen_wxai](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/trossen_wxai), scene `scene.xml`.
