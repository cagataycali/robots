---
description: Nine short chapters, one arm to a recorded multi-robot dataset.
---

# Tutorial

```bash
pip install "strands-robots[sim-mujoco]"   # chapters 1–6
pip install "strands-robots[all]"          # all chapters
pip install strands-agents                  # chapter 4
```

| # | Chapter | Covers |
|---|---------|--------|
| 1 | [Your first robot](01-your-first-robot.md) | `Robot("so100")`, `step`, `render` |
| 2 | [Simulation](02-simulation.md) | cameras, objects, randomize, physics |
| 3 | [Policies](03-policies.md) | Mock / GR00T / LeRobot / Cosmos3, `create_policy` |
| 4 | [AI agents](04-agents.md) | `Agent(tools=[robot])`, natural-language control |
| 5 | [Multi-robot](05-multi-robot.md) | Zenoh mesh, peer RPC, emergency stop |
| 6 | [Recording](06-recording.md) | LeRobot v3 dataset, `start_recording` / `stop_recording` |
| 7 | [Training](07-training.md) | LeRobot / GR00T / Cosmos upstream pipelines |
| 8 | [Real hardware](08-real-hardware.md) | `mode="real"`, calibration, cameras, teleop |
| 9 | [Advanced](09-advanced.md) | factory internals, custom backends, lazy imports |

Blocks marked `# requires hardware` need a servo controller; `# requires GPU` need CUDA + a checkpoint. Everything else runs on laptop CPU.

## See also

- [Learning path](../learning-path.md) — pick the right track.
- [Quickstart](../getting-started/quickstart.md) — condensed to one page.
- [Architecture](../architecture.md) — module map referenced throughout.
