# Installation

## Requirements

- **Python 3.12+** (3.12, 3.13, 3.14)
- **pip** (comes with Python)

## Basic Install

```bash
pip install strands-robots
```

This gives you the core library — robot registry, factory, policy abstraction. Core dependencies: `strands-agents`, `numpy`, `opencv-python-headless`, `Pillow`, `msgpack`, `pyzmq`.

> **Note**: Python 3.12+ is required. Python 3.10 and 3.11 are no longer supported.

## With Extras

Install what you need:

```bash
pip install "strands-robots[mujoco]"           # MuJoCo simulation
pip install "strands-robots[lerobot]"          # LeRobot + Feetech servos
pip install "strands-robots[isaac]"            # Isaac Sim/Lab (Linux + NVIDIA GPU)
pip install "strands-robots[newton]"           # Newton GPU physics (Linux + NVIDIA GPU)
pip install "strands-robots[cosmos-transfer]"  # Sim→Real visual transfer
pip install "strands-robots[cosmos-predict]"   # World model policy
pip install "strands-robots[zenoh]"            # P2P robot mesh networking
pip install "strands-robots[policies]"         # All policy dependencies
pip install "strands-robots[all]"              # Everything
```

## From Source

```bash
git clone https://github.com/strands-labs/robots
cd robots
pip install -e ".[all]"
```

## Verify

```python
from strands_robots import Robot, list_robots

# List all available robots
robots = list_robots()
print(f"{len(robots)} robots available")

# Create a sim robot
robot = Robot("so100")
print(robot)  # MujocoBackend for SO-100
```

If this runs without errors, you're ready.

## MuJoCo Interactive Viewer (macOS)

If you want to use MuJoCo's interactive 3D viewer window on macOS, use `mjpython`:

```bash
# mjpython is installed with the mujoco package
pip install mujoco

# Run your script with mjpython instead of python
mjpython my_script.py
```

This is only needed for the interactive viewer window. Headless rendering (`render()`) works with standard `python` on all platforms. On Linux and Windows, both headless and interactive rendering work with `python`.

## Strands Agents

Strands Robots uses [Strands Agents](https://github.com/strands-agents/sdk-python) for natural language control. It's included as a dependency — no separate install needed.

For the default Bedrock model provider, configure AWS credentials. See the [Strands quickstart](https://strandsagents.com/) for other providers (Ollama, OpenAI, Anthropic, etc.).
