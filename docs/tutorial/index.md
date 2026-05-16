---
description: Nine chapters from "hello world" to "fleet on the mesh". Each chapter is 10–20 minutes.
---

# Tutorial

A guided walk through everything `strands-robots` can do. Each chapter builds on the
previous one. Skip ahead if a step is obvious — the chapters are self-contained.

## What you'll build

By the end, you will have:

- Spawned a simulated arm and rendered a frame.
- Loaded a real-world MJCF scene with two robots and a cube.
- Plugged in three different VLA policies (Mock, GR00T, LeRobot) without changing user code.
- Wired a Strands `Agent` to control the robot via natural language.
- Coordinated two robots over a Zenoh mesh.
- Recorded a LeRobot v3 dataset.
- Opened a sim → real path with a calibrated SO-100 arm and live camera streams.

## Chapters

| # | Chapter | What it covers | Time |
|---|---------|----------------|------|
| 1 | [Your first robot](01-your-first-robot.md) | `Robot("so100")`, `step`, `render`, `list_robots()`. | 10 min |
| 2 | [Simulation](02-simulation.md) | `Simulation` actions: scenes, cameras, objects, randomization. | 15 min |
| 3 | [Policies](03-policies.md) | The `Policy` ABC, `MockPolicy` / `Gr00tPolicy` / `LerobotLocalPolicy`, `create_policy()`. | 15 min |
| 4 | [AI agents](04-agents.md) | `Agent(tools=[robot])`, natural-language control, `agent("pick up the cube")`. | 15 min |
| 5 | [Multi-robot](05-multi-robot.md) | Two `Robot()` instances on the Zenoh mesh, peer discovery, RPC. | 15 min |
| 6 | [Recording data](06-recording.md) | `start_recording` / `stop_recording`, LeRobot v3 dataset structure. | 10 min |
| 7 | [Training](07-training.md) | What ships, what doesn't, links to LeRobot/GR00T training pipelines. | 10 min |
| 8 | [Real hardware](08-real-hardware.md) | `mode="real"`, calibration, camera mapping, teleop, safety defaults. | 20 min |
| 9 | [Advanced](09-advanced.md) | Custom backends, custom data_configs, factory internals, lazy imports. | 15 min |

## Conventions

Every code block in this tutorial works against the current `main` branch. Where a step
needs hardware or a GPU we'll mark it explicitly:

- `# requires hardware` — needs a real servo controller plugged in
- `# requires GPU` — needs CUDA + a model checkpoint
- everything else runs on a laptop CPU

## Setup before chapter 1

```bash
# Minimum for the tutorial
pip install "strands-robots[sim-mujoco]"

# Optional but recommended
pip install "strands-robots[all]"
```

You'll also want a Strands Agents install for chapter 4:

```bash
pip install strands-agents
```

## See also

- [Learning path](../learning-path.md) — pick the right track if you don't want to do the
  tutorial in order.
- [Quickstart](../getting-started/quickstart.md) — same idea, condensed to one page.
- [Architecture](../architecture.md) — the module map you'll see referenced throughout.
