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
robot.add_object(name="red_cube", shape="box", size=[0.05, 0.05, 0.05],
                 position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0])  # the arm faces -Y
robot.add_camera(name="front", position=[0.3, -0.7, 0.45], target=[0.0, -0.2, 0.03])
Agent(tools=[robot])("pick up the red cube")
```

In MuJoCo the SO-100 jaw cannot squeeze the cube hard enough to lift it, so the
agent carries it with `attach_bodies(mode="weld")`, a grasp assist that
`set_gripper` names as soon as the jaw closes.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "strands-robots[sim-mujoco]"   # plain pip works too
```

Python 3.12+. Everything else is an extra you pull in when you need it -
`lerobot` (hardware, local VLA inference, recording), `groot` (GR00T N1.7),
`cosmos3-service`, `mesh`, `mesh-iot`, `sim-newton`, `sim-isaac`, `wbc` -
see [Installation](https://strands-labs.github.io/robots/start/install/) for the full table.

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
| **150+ robots across 8 categories** - arms, bimanual, hands and grippers, humanoids, mobile bases, mobile manipulators, aerial, expressive - from one registry with asset auto-download | [Robots](https://strands-labs.github.io/robots/robots/) |
| **Any policy** behind one ABC: LeRobot (ACT / Pi0 / SmolVLA / Diffusion / GR00T N1.7), Cosmos 3, MolmoAct2, whole-body control, cuRobo, MoveIt2, scripted | [Policies](https://strands-labs.github.io/robots/learn/policies/) |
| **Teleoperate and record** LeRobotDataset episodes from leader arms, gamepads or WASD; stream to HF datasets or Storage Buckets | [Teleoperation](https://strands-labs.github.io/robots/learn/hardware/teleoperation/), [Recording](https://strands-labs.github.io/robots/learn/data/record/) |
| **Train** with LeRobot (ACT to GR00T N1.7), Cosmos 3 or RL (PPO / FastSAC), locally or as a SageMaker job, then run the checkpoint in sim and on hardware | [Training](https://strands-labs.github.io/robots/learn/training/lerobot/) |
| **Simulate** with an agent-callable MuJoCo tool: worlds, terrain, domain randomization, rendering, dataset capture; Newton and Isaac backends | [Simulation](https://strands-labs.github.io/robots/learn/simulation/) |
| **Mesh** every `Robot(mesh=True)` as a Zenoh peer (on one machine set `STRANDS_MESH_LOCAL_DEV=true`; across hosts, mTLS and an ACL): `robot.mesh.tell(peer, instruction, policy_provider=...)` asks another robot to run a policy; broadcast an E-STOP, bridge fleets over AWS IoT Core | [Mesh](https://strands-labs.github.io/robots/learn/mesh/fleet/) |
| **ROS 2** - observe and command any graph (`use_ros`), act as a node without rclpy (`use_rtps`), expose a running sim | [ROS 2](https://strands-labs.github.io/robots/learn/ros2/) |
| **Configure** every environment variable the package reads, with its default and its guard | [Configuration](https://strands-labs.github.io/robots/reference/configuration/) |

<p align="center">
  <img src="docs/assets/mesh_network.svg" alt="Strands Robots mesh: every Robot(mesh=True) is a Zenoh peer; an agent lists peers, tells one what to do, and an emergency stop reaches all of them" width="100%">
</p>

Real servos never move by accident: `mode="real"` is an explicit opt-in.

## Documentation

Full guide, API reference and per-robot pages:
**[strands-labs.github.io/robots](https://strands-labs.github.io/robots)** -
start with the [Quickstart](https://strands-labs.github.io/robots/start/first-robot/) and [Architecture](https://strands-labs.github.io/robots/concepts/architecture/).

## Development

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all,dev]" hatch   # hatch is the task runner, in no extra
hatch run test && hatch run lint       # pytest; ruff + mypy
```

Conventions and review learnings are in [AGENTS.md](AGENTS.md);
[CONTRIBUTING](https://strands-labs.github.io/robots/reference/project/contributing/) covers the workflow. Work is tracked on the
[project board](https://github.com/orgs/strands-labs/projects/2).

## Security

Found a vulnerability? **Do not** open a public issue - follow
[SECURITY.md](SECURITY.md). The `trust_remote_code` gate on `lerobot_local`, the
mesh CA-pinning controls and the ordered
[CA Pin Rotation Runbook](https://strands-labs.github.io/robots/reference/configuration/#ca-pin-rotation-runbook)
are documented in the [Configuration](https://strands-labs.github.io/robots/reference/configuration/) matrix.

## License

Apache-2.0 - see [LICENSE](LICENSE).
