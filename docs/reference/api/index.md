---
description: The public Python surface of strands_robots at this commit, rendered from the docstrings, one page per group.
---

# API reference

The public Python surface of `strands_robots` at this commit, rendered from the docstrings. Every name below is importable from the top-level package (`strands_robots.__all__`) or from the sub-package the page names. After this page you know which page holds the symbol you need.

| page | what it covers | import from |
|---|---|---|
| [Robot and factory](robot.md) | `Robot(...)` factory, the hardware `Robot` class, `Teleoperator` | `strands_robots` |
| [Registry](registry.md) | `list_robots`, `get_robot`, `resolve_name`, discovery, user robots | `strands_robots.registry` |
| [Simulation](simulation.md) | `create_simulation`, `SimEngine`, `SimWorld`, `SimRobot`, `SimObject`, `SimCamera`, backends | `strands_robots.simulation` |
| [Policies](policies.md) | `Policy`, `create_policy`, `register_policy`, `MockPolicy`, `Gr00tPolicy`, persistent cache | `strands_robots.policies` |
| [Drivers](drivers.md) | `HardwareDriver` protocol, native driver registry | `strands_robots.drivers` |
| [Tools](tools.md) | the `@tool` functions re-exported at the top level | `strands_robots` |
| [Data](data.md) | streaming datasets, bucket sync, episode judging | `strands_robots` |
| [Mesh](mesh.md) | `Mesh`, `init_mesh`, sessions, peers, ROS and RTPS bridges, device connect | `strands_robots.mesh` |

Heavy names are lazy: `import strands_robots` does not import torch, lerobot, numpy or mujoco. The first attribute access (`strands_robots.Robot`) imports for real and raises `AttributeError` naming the missing dependency when an extra is not installed.

mkdocstrings generates the pages from the source by static analysis, so they cannot drift from the code; signatures show the code's defaults. The tool catalog with action lists is [Tools](../tools.md); environment variables are [Configuration](../configuration.md).
