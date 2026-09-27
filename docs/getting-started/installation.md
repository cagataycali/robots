---
description: Install strands-robots with uv - extras matrix, platform notes, headless rendering.
---

# Installation

Requires **Python >= 3.12**. Examples use [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`); plain `pip install` works too.

`uv pip install` installs into the active virtual environment and refuses when there is none (`No virtual environment found; run uv venv`), so create and activate one first:

```bash
uv venv --python 3.12
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## Extras matrix

| Extra | Pulls in | When you need it |
|-------|----------|------------------|
| (none) | core only - Robot factory, registry, lazy imports | Inspect the catalog, write tools |
| `[sim]` | `robot_descriptions>=1.23.0,<2.0.0` | Sim asset resolution without MuJoCo |
| `[sim-mujoco]` | `sim` + `mujoco`, `imageio`, `imageio-ffmpeg` | Any `Robot()` with default `mode="sim"` |
| `[lerobot]` | `lerobot>=0.6.1,<0.7.0`, `psutil>=6.0.0,<8.0.0` | `LerobotLocalPolicy` + dataset recording + the `lerobot_train` / `lerobot_teleoperate` session tools |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` (ZMQ to a GR00T container) |
| `[cosmos3-service]` | `msgpack`, `websockets>=17.0` | `Cosmos3Policy` (WebSocket to Cosmos 3 server) |
| `[earthrover]` | `requests>=2.28.0,<3.0.0` | `Robot("earthrover", mode="real", driver="strands")` - HTTP to the earth-rovers-sdk |
| `[ur]` | `ur-rtde>=1.6.0,<2.0.0` | `Robot("ur5e", mode="real", driver="strands")` - RTDE to a UR controller |
| `[rl]` | `sim-mujoco` + `torch>=2.0`, `gymnasium>=0.29,<2.0` | From-scratch RL: `create_trainer("ppo")`, `FastSacTrainer`, `FastTd3Trainer`, `GymSimEnv` |
| `[mesh]` | `eclipse-zenoh>=1.6.1,<2.0.0`, `json5` | Multi-robot mesh discovery + RPC |
| `[mesh-iot]` | `mesh` + `awsiotsdk`, `awscrt`, `boto3` | AWS IoT Core transport for mesh |
| `[all]` | 21 of the 33 extras - **not** a union. `[cosmos3-diffusers]`, `[cosmos3-service]`, `[cosmos3-sim]`, `[crazyflie]` (GPLv3), `[curobo]`, `[microduck]`, `[ros2]`, `[sim-gs]`, `[sim-isaac]`, `[sim-newton]` and `[ur]` (compiled binding) stay opt-in | Demos, CI, exploration |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pytest-timeout` | Contributing |

```bash
# inside the activated venv from above
uv pip install "strands-robots[sim-mujoco]"                  # sim only
uv pip install "strands-robots[all]"                         # the 21-extra bundle
uv pip install "strands-robots[sim-mujoco,cosmos3-service]"  # Cosmos 3
uv pip install "strands-robots[sim-mujoco,lerobot,mesh]"     # pick and choose
```

Each extra's floor is the oldest release the code was measured against; the reason for each is in `pyproject.toml`.

## Platform notes

**macOS:** works out of the box (arm64 + Intel).

**Linux (headless / real hardware):**
```bash
sudo apt install libosmesa6-dev ffmpeg
sudo usermod -aG dialout $USER   # USB serial access; re-login after
```

**Windows:** WSL2 + Ubuntu 22.04 (native Windows works for sim, not actively tested).

**Jetson / aarch64 (JetPack):**
```bash
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

Do not pin `numpy < 2` first: `lerobot >= 0.6` requires `numpy >= 2`, and JetPack's torch runs on it.

For CUDA torch on Jetson, let `uv` pick NVIDIA's wheels:

```bash
export UV_TORCH_BACKEND=auto   # resolves +cu130 wheels for Thor/Jetson
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

### MolmoAct2 on Jetson

Details on [LeRobot Local: MolmoAct2](../reference/policies/lerobot-local.md#molmoact2).

```bash
# The [molmoact2] extra layers transformers, peft, scipy on top of lerobot >= 0.6;
# lerobot 0.6 pulls the aarch64 torchcodec decoder itself:
uv pip install "strands-robots[molmoact2]"
```

## Headless rendering

```bash
export MUJOCO_GL=osmesa     # software rendering - Linux
export MUJOCO_GL=egl        # hardware EGL
```

## Verify

`doctor` checks this machine the way the runtime will read it - the interpreter
and package, each extra, the GL backend, the torch/torchcodec pair, the GPU, the
serial and Hub credentials, and the device-connect and mesh postures - and exits
non-zero if any row fails:

```bash
python -m strands_robots doctor           # run every check
python -m strands_robots doctor --list    # print the check names, probe nothing
```

```python
from strands_robots import Robot

sim = Robot("so100")
sim.step()
obs = sim.get_observation("so100")
# obs is a flat dict mixing per-joint state floats and per-camera ndarrays:
#   {'shoulder_pan.pos': 0.0, ..., 'gripper.pos': 0.0, 'default': <HxWx3 uint8>}
print(list(obs.keys()))
```

Assets cache under `~/.strands_robots/assets/`.

## See also

- [Quickstart](quickstart.md) - five minutes after install.
- [Robot factory](robot-factory.md) - every kwarg `Robot()` accepts.
- [Troubleshooting](../reference/troubleshooting.md) - install gotchas.
- [Configuration](../reference/configuration.md) - every environment variable.
