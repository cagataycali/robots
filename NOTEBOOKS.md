# 📓 Strands Robots — Notebook Guide

Educational Jupyter notebooks for learning and using the `strands-robots` library.
Each notebook is self-contained with clear explanations, runnable code, and expected outputs.

## Notebooks

| # | Notebook | Topics | Prerequisites |
|---|----------|--------|---------------|
| 1 | [Getting Started](notebooks/01_getting_started.ipynb) | Installation, imports, lazy loading, package structure overview | None |
| 2 | [Policy Abstraction](notebooks/02_policy_abstraction.ipynb) | Policy ABC, MockPolicy, create_policy() factory, custom policies, sync/async | None |
| 3 | [GR00T Policy Deep Dive](notebooks/03_groot_policy.ipynb) | Gr00tDataConfig, data_configs.json, observation/action mappings, ZMQ client, local vs service mode | `pip install strands-robots[groot-service]` |
| 4 | [LeRobot Local Inference](notebooks/04_lerobot_local.ipynb) | LerobotLocalPolicy, HF model loading, ProcessorBridge, VLA tokenization, Real-Time Chunking (RTC) | `pip install strands-robots[lerobot]` (GPU recommended) |
| 5 | [MuJoCo Simulation](notebooks/05_mujoco_simulation.ipynb) | create_simulation(), world lifecycle, add robots/objects, physics stepping, rendering, cameras | `pip install strands-robots[sim-mujoco]` |
| 6 | [Simulation — Policy Runner & Recording](notebooks/06_sim_policy_and_recording.ipynb) | run_policy() in sim, eval_policy(), domain randomization, trajectory recording to LeRobotDataset | `pip install strands-robots[all]` |
| 7 | [Robot Tool & Agent Integration](notebooks/07_robot_agent_tool.ipynb) | Robot AgentTool, Strands Agent, natural language control, async execution, status/stop | LeRobot hardware or mock |
| 8 | [Registry & Model Resolution](notebooks/08_registry_and_models.ipynb) | robots.json, policies.json, resolve_name(), resolve_model(), asset manager, custom registration | None |

## Quick Start

```bash
# Install base package
pip install strands-robots

# For simulation notebooks (5, 6)
pip install "strands-robots[sim-mujoco]"

# For all notebooks
pip install "strands-robots[all]"
```

## Architecture Reference

```
strands_robots/
├── policies/              # Notebook 2, 3, 4
│   ├── base.py            # Policy ABC (async get_actions + sync wrapper)
│   ├── factory.py         # create_policy() + registry integration
│   ├── mock.py            # MockPolicy (sinusoidal test actions)
│   ├── groot/             # Notebook 3
│   │   ├── policy.py      # Gr00tPolicy (service + local N1.5/N1.6)
│   │   ├── client.py      # ZMQ inference client
│   │   └── data_config.py # Embodiment configs + _extends inheritance
│   └── lerobot_local/     # Notebook 4
│       ├── policy.py      # Direct HF inference (ACT, Pi0, SmolVLA, ...)
│       ├── processor.py   # Pre/post processor pipeline bridge
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
├── simulation/            # Notebooks 5, 6
│   ├── base.py            # SimEngine ABC
│   ├── factory.py         # create_simulation() + backend registration
│   ├── models.py          # SimWorld, SimRobot, SimObject dataclasses
│   ├── model_registry.py  # URDF/MJCF resolution
│   └── mujoco/            # MuJoCo backend (mixins)
│       ├── simulation.py  # Simulation AgentTool (orchestrator)
│       ├── physics.py     # Raycasting, Jacobians, forces, checkpoints
│       ├── rendering.py   # Camera rendering, observations
│       ├── policy_runner.py # run_policy, eval_policy, replay
│       ├── randomization.py # Domain randomization
│       ├── recording.py   # LeRobotDataset recording
│       ├── mjcf_builder.py # MJCF XML builder
│       └── scene_ops.py   # XML inject/eject
├── robot.py               # Notebook 7 — Robot AgentTool
├── registry/              # Notebook 8
│   ├── robots.py          # Robot query/resolve
│   ├── policies.py        # Policy resolve/import
│   └── loader.py          # JSON hot-reload
├── tools/                 # Strands @tool functions
│   ├── gr00t_inference.py # Docker service management
│   ├── pose_tool.py       # Pose store/load/interpolate
│   └── ...
├── utils.py               # require_optional(), path resolution
└── dataset_recorder.py    # LeRobotDataset bridge
```

## Conventions

- All notebooks use `strands_robots` import paths (not internal modules)
- GPU-requiring cells are clearly marked with `# Requires: GPU`
- Each notebook can be run independently (no cross-notebook dependencies)
- Mock/test alternatives are provided for hardware-dependent operations
