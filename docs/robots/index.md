---
title: Robots
description: Every robot strands-robots knows by name, filterable by family, each with a live 3D viewer and the one line that builds it.
---

# Robots

You have every robot the registry knows, all {{n:robots}} of them across {{n:categories}} families, on one page: filter by family, open a card for the robot's own page with the 3D viewer, the one-line constructor, the aliases `Robot()` accepts, and the hardware facts read from its driver. {{n:sim_assets}} of them load in MuJoCo from a pinned public model; the rest are hardware definitions only. Everything here is generated from `strands_robots/registry/robots.json` at build time, so a robot added to the registry appears at the next build and a stale row cannot exist.

```python
from strands_robots import Robot

robot = Robot("so101")  # simulation, the default
```

The same call with `mode="real"` returns the object that drives the hardware; it needs the arm on a serial port:

```python title="sketch"
arm = Robot("so101", mode="real", port="/dev/ttyACM0")
```

<div class="sr-filter" role="group" aria-label="Filter robots by family" markdown="0">
  <button type="button" class="sr-filter-btn" data-family="all" aria-pressed="true">All</button>
  <button type="button" class="sr-filter-btn" data-family="arm" aria-pressed="false">Arms</button>
  <button type="button" class="sr-filter-btn" data-family="bimanual" aria-pressed="false">Bimanual</button>
  <button type="button" class="sr-filter-btn" data-family="hand" aria-pressed="false">Hands and grippers</button>
  <button type="button" class="sr-filter-btn" data-family="humanoid" aria-pressed="false">Humanoids</button>
  <button type="button" class="sr-filter-btn" data-family="mobile" aria-pressed="false">Mobile bases</button>
  <button type="button" class="sr-filter-btn" data-family="mobile_manip" aria-pressed="false">Mobile manipulators</button>
  <button type="button" class="sr-filter-btn" data-family="aerial" aria-pressed="false">Aerial</button>
  <button type="button" class="sr-filter-btn" data-family="expressive" aria-pressed="false">Expressive</button>
</div>

{{robot_cards}}

## Coverage

One row per robot: which `driver=` builds it for real, the asset directory its simulation loads, and the policy providers written for that body. Read from the driver table, the registry and each provider's own module at this commit.

{{coverage_matrix}}
