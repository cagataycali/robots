---
description: Visual roadmap from "first import" to "fleet on the mesh" — pick your level.
---

# Learning Path

Three tracks, one repo.

## Track 1 — "I just want to see it move"

```mermaid
graph LR
    A[Install] --> B[Robot('so100')]
    B --> C[render frame]
    C --> D[run a policy]
    D --> E[Agent('pick up the cube')]
    classDef step fill:#2ea44f,stroke:#1b7735,color:#fff
    class A,B,C,D,E step
```

| Step | Page | Time |
|------|------|------|
| 1 | [Installation](getting-started/installation.md) | 2 min |
| 2 | [Quickstart](getting-started/quickstart.md) | 5 min |
| 3 | [Tutorial 1 — Your first robot](tutorial/01-your-first-robot.md) | 10 min |
| 4 | [Tutorial 2 — Simulation](tutorial/02-simulation.md) | 15 min |
| 5 | [Tutorial 4 — AI agents](tutorial/04-agents.md) | 15 min |

## Track 2 — "I want to ship something with this"

```mermaid
graph LR
    A[Robot Factory] --> B[Pick a Policy]
    B --> C[Record dataset]
    C --> D[Hardware bring-up]
    D --> E[Multi-robot]
    classDef step fill:#0969da,stroke:#044289,color:#fff
    class A,B,C,D,E step
```

| Step | Page | What you walk away with |
|------|------|-------------------------|
| 1 | [Robot factory](getting-started/robot-factory.md) | Full `Robot(...)` signature |
| 2 | [Policy providers](policies/overview.md) → [GR00T](policies/groot.md), [LeRobot](policies/lerobot-local.md), or [Cosmos 3](policies/cosmos3.md) | Working VLA policy |
| 3 | [Tutorial 6 — Recording](tutorial/06-recording.md) → [Recording reference](recording.md) | LeRobot v3 dataset on disk |
| 4 | [Tutorial 8 — Real hardware](tutorial/08-real-hardware.md) → [Hardware tools](hardware/tools.md) | Real arm calibrated + cameras |
| 5 | [Tutorial 5 — Multi-robot](tutorial/05-multi-robot.md) | Two `Robot()` instances on the mesh |

## Track 3 — "I want to extend the library"

```mermaid
graph LR
    A[Architecture] --> B[Custom Policies]
    B --> C[Custom Backend]
    C --> D[Add a Robot]
    D --> E[Open a PR]
    classDef step fill:#8250df,stroke:#5a32a3,color:#fff
    class A,B,C,D,E step
```

| Step | Page | What you'll touch |
|------|------|-------------------|
| 1 | [Architecture](architecture.md) | Module boundaries, ABC contracts |
| 2 | [Custom policies](policies/custom-policies.md) | `policies/base.py`, factory, registry |
| 3 | [Tutorial 9 — Advanced](tutorial/09-advanced.md) | `SimEngine` ABC, data_configs, tool authoring |
| 4 | [Robot catalog](robots/index.md) | `registry/robots.json` schema |
| 5 | [Contributing](contributing.md) | hatch envs, lint, PR conventions |

## Quick references

[API reference](api-reference.md) · [Robot catalog](robots/index.md) · [Troubleshooting](troubleshooting.md) · [Examples](examples/overview.md)
