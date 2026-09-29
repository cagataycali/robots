---
title: reachy_mini
description: "Pollen Reachy Mini (6-DOF Stewart head + antennas, 9 actuators)"
---

<!-- generated: docs/hooks/robot_pages.py -->

# Pollen Reachy Mini (6-DOF Stewart head + antennas, 9 actuators)

{{robot_chips:reachy_mini}}

<robot-viewer name="reachy_mini"></robot-viewer>

```python
from strands_robots import Robot

robot = Robot("reachy_mini")
```

```python title="sketch"
robot = Robot("reachy_mini", mode="real", port="reachy-mini.local:8000")  # ReachyDriver
```

Aliases: `pollen_reachy_mini`, `reachy`, `reachy-mini`, `reachymini`.

## Hardware

**`ReachyDriver`** (the default for this robot) speaks Reachy daemon REST API plus its real-time link: [port, SDK, kwargs and checks](../learn/hardware/drivers.md#reachydriver).

## Policies verified on this robot

No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then [train](../learn/training/lerobot.md) and run it with `run_policy`.

Model: [pollen-robotics/reachy_mini/src/reachy_mini/descriptions/reachy_mini](https://github.com/pollen-robotics/reachy_mini/tree/292b2434cadbb3ff932863bd9b476741bb6ef2fd/src/reachy_mini/descriptions/reachy_mini), scene `mjcf/scene.xml`.
