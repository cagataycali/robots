<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://strandsagents.com/latest/assets/wordmark-github-dark.svg">
        <img src="https://strandsagents.com/latest/assets/wordmark-github-light.svg" alt="Strands" width="320">
      </picture>
    </a>
  </div>

  <h1>
    Strands Robots
  </h1>

  <h2>
    Control, simulate, and train robots with natural language
  </h2>

  <div align="center">
    <a href="https://pypi.org/project/strands-robots/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/strands-robots"/></a>
    <a href="https://github.com/strands-labs/robots"><img alt="GitHub stars" src="https://img.shields.io/github/stars/strands-labs/robots"/></a>
    <a href="https://github.com/strands-labs/robots/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-labs/robots"/></a>
    <a href="https://github.com/google-deepmind/mujoco"><img alt="MuJoCo" src="https://img.shields.io/badge/MuJoCo-3.x-000000"/></a>
    <a href="https://github.com/NVIDIA/Isaac-GR00T"><img alt="GR00T" src="https://img.shields.io/badge/NVIDIA-GR00T-76B900?logo=nvidia"/></a>
    <a href="https://github.com/huggingface/lerobot"><img alt="LeRobot" src="https://img.shields.io/badge/🤗-LeRobot-yellow"/></a>
  </div>

  <p>
    <a href="https://strandsagents.com/">Strands Docs</a>
    ◆ <a href="https://github.com/google-deepmind/mujoco">MuJoCo</a>
    ◆ <a href="https://github.com/NVIDIA/Isaac-GR00T">NVIDIA GR00T</a>
    ◆ <a href="https://github.com/huggingface/lerobot">LeRobot</a>
    ◆ <a href="https://github.com/orgs/strands-labs/projects/2">Project Board</a>
  </p>
</div>

<p align="center">
  <img src="docs/assets/hero_loop.svg" alt="Strands Robots: one Robot object, any robot. A Strands Agent perceives, reasons and acts; the world, MuJoCo or a real arm, answers" width="100%">
</p>

`strands-robots` gives a [Strands Agent](https://github.com/strands-agents/harness-sdk)
hands. One `Robot()` call returns a **MuJoCo simulation** (default: no GPU, no
hardware) or a **real robot** - same code, same natural-language control, same
opt-in peer-to-peer **mesh**. Learned policies from the Hugging Face Hub, from
vision-language-action models to world foundation models and whole-body
controllers, run through the same `run_policy` call in the simulator and on
the physical robot.

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")              # MuJoCo sim by default; mode="real" for hardware
robot(action="add_object", name="red_cube", shape="box",
      size=[0.025, 0.025, 0.025], position=[0.0, 0.20, 0.025],
      color=[1.0, 0.0, 0.0, 1.0])
robot(action="add_camera", name="front", position=[0.0, 0.5, 0.3],
      target=[0.0, 0.20, 0.025])
Agent(tools=[robot])("pick up the red cube")
```

A fresh `Robot("so100")` scene is the arm alone on a ground plane with one
external camera; the two `action=` calls above put a red cube on the open
`+Y` side of the base (the gripper at zero pose extends toward `-Y`, so this
side is unoccluded) and add a `front` camera whose line of sight to the cube
does not pass through the gripper, so a language-conditioned policy has
something to see. [First robot](docs/start/first-robot.md) walks through the
one-robot flow end-to-end.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "strands-robots[sim-mujoco]"   # plain pip works too
```

Python 3.12+. Everything else is an extra you pull in when you need it -
`lerobot` (hardware, local VLA inference, recording), `groot` (GR00T N1.7),
`cosmos3-service`, `mesh`, `mesh-iot`, `sim-newton`, `sim-isaac`, `wbc` -
see [Installation](docs/start/install.md) for the full table.

## How it works

<p align="center">
  <img src="docs/assets/architecture_flow.svg" alt="How strands-robots is layered: Strands Agent, policies, backends, robots; actions flow down, observations flow up" width="100%">
</p>

A prompt reaches the agent; the agent calls the robot tool; a policy turns the
observation into an action chunk; the backend (MuJoCo, Newton, Isaac, or the
hardware driver) executes it and returns the next observation. Sim and hardware
share the policy interface, the mesh, and the tool surface, so a workflow proven
in sim runs on the metal by changing `mode`.

## What you get

| | Read |
|---|---|
| **150+ robots across 8 categories** - arms, bimanual rigs, humanoids, quadrupeds, hands, drones - from one registry with asset auto-download | [Robots](docs/robots/index.md) |
| **Any policy** behind one ABC: LeRobot (ACT / Pi0 / SmolVLA / Diffusion / GR00T N1.7), Cosmos 3, MolmoAct2, whole-body control, cuRobo, MoveIt2, scripted | [Policies](docs/learn/policies/index.md) |
| **Teleoperate and record** LeRobotDataset episodes from leader arms, gamepads or WASD; stream to HF datasets or Storage Buckets | [Teleoperation](docs/learn/hardware/teleoperation.md), [Recording](docs/learn/data/record.md) |
| **Train** with LeRobot (ACT to GR00T N1.7), Cosmos 3 or RL (PPO / FastSAC), locally or as a SageMaker job, then run the checkpoint in sim and on hardware | [Training](docs/learn/training/lerobot.md) |
| **Simulate** with an agent-callable MuJoCo tool: worlds, terrain, domain randomization, rendering, dataset capture; Newton and Isaac backends | [Simulation](docs/learn/simulation/index.md) |
| **Mesh** every robot as a Zenoh peer: `tell()` another robot what to do, broadcast an E-STOP, bridge fleets over AWS IoT Core | [Mesh](docs/learn/mesh/fleet.md) |
| **ROS 2** - observe and command any graph (`use_ros`), act as a node without rclpy (`use_rtps`), expose a running sim | [ROS 2](docs/learn/ros2.md) |
| **Configure** every environment variable the package reads, with its default and its guard | [Configuration](docs/reference/configuration.md) |

<p align="center">
  <img src="docs/assets/mesh_network.svg" alt="Strands Robots mesh: every Robot(mesh=True) is a Zenoh peer; an agent lists peers, tells one what to do, and an emergency stop reaches all of them" width="100%">
</p>

Real servos never move by accident: `mode="real"` is an explicit opt-in.

## Documentation

Full guide, API reference and per-robot pages:
**[strands-labs.github.io/robots](https://strands-labs.github.io/robots)** -
start with the [Quickstart](docs/start/first-robot.md) and [Architecture](docs/concepts/architecture.md).

## Development

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all,dev]"
hatch run test && hatch run lint   # pytest; ruff + mypy
```

Conventions and review learnings are in [AGENTS.md](AGENTS.md);
[CONTRIBUTING](docs/reference/project/contributing.md) covers the workflow. Work is tracked on the
[project board](https://github.com/orgs/strands-labs/projects/2).

## Security

Found a vulnerability? **Do not** open a public issue - follow
[SECURITY.md](SECURITY.md). The `trust_remote_code` gate on `lerobot_local`, the
mesh CA-pinning controls and the ordered
[CA Pin Rotation Runbook](docs/reference/configuration.md#ca-pin-rotation-runbook)
are documented in the [Configuration](docs/reference/configuration.md) matrix.

## License

Apache-2.0 - see [LICENSE](LICENSE).
