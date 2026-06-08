---
description: pip install strands-robots — extras matrix, platform notes, headless rendering.
---

# Installation

`strands-robots` requires **Python ≥ 3.12**. It is a single `pip install`. Pick the
extras that match your use case.

## Extras matrix

| Extra | Pulls in | When you need it |
|-------|----------|------------------|
| (none) | core only — Robot factory, registry, lazy imports | Inspect the catalog, write tools |
| `[sim]` | `robot_descriptions` | Sim asset resolution without MuJoCo |
| `[sim-mujoco]` | `sim` + `mujoco`, `imageio`, `imageio-ffmpeg` | Any `Robot()` with default `mode="sim"` |
| `[lerobot]` | `lerobot>=0.5.0,<0.6.0` | `LerobotLocalPolicy` + dataset recording |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` (talks to a GR00T container over ZMQ) |
| `[cosmos3-service]` | `msgpack`, `websockets` | `Cosmos3Policy` (talks to a Cosmos 3 server over WebSocket) |
| `[mesh]` | `eclipse-zenoh`, `json5` | Multi-robot mesh discovery + RPC |
| `[mesh-iot]` | `mesh` + `awsiotsdk`, `awscrt`, `boto3` | AWS IoT Core transport for mesh |
| `[benchmark-libero]` | `libero` eval deps | LIBERO benchmark suite |
| `[all]` | `groot-service` + `lerobot` + `sim-mujoco` + `mesh` + `mesh-iot` | Demos, CI, exploration |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pytest-timeout` | Contributing |

## Common flavours

### "I just want simulation"

```bash
pip install "strands-robots[sim-mujoco]"
```

CPU-only MuJoCo. Works on a laptop. Covers tutorial chapters 1-7.

### "I want everything"

```bash
pip install "strands-robots[all]"
```

Sim + LeRobot + GR00T + mesh + benchmarks. Big install but no surprises.

### "Real hardware only"

```bash
pip install "strands-robots[lerobot]"
```

LeRobot + dataset recording. Skip `[sim-mujoco]` if you'll never simulate.

### "Cosmos 3 inference"

```bash
pip install "strands-robots[sim-mujoco,cosmos3-service]"
```

Connects to an NVIDIA Cosmos 3 action-policy server over WebSocket. See
[Cosmos3Policy](../policies/cosmos3.md).

### "Custom — pick what you need"

```bash
pip install "strands-robots[sim-mujoco,lerobot,mesh]"
```

## Platform notes

### macOS (Apple Silicon / Intel)

Works out of the box. Use the system Python or `pyenv`. MuJoCo and LeRobot ship native
arm64 wheels.

```bash
pip install "strands-robots[all]"
```

### Linux

You may want a couple of system packages for headless rendering and video export:

```bash
sudo apt install libosmesa6-dev ffmpeg
pip install "strands-robots[all]"
```

For real hardware, add the user to the `dialout` group so USB serial devices are
accessible without sudo:

```bash
sudo usermod -aG dialout $USER
# log out and back in
```

### Windows

WSL2 is the supported path. Install Ubuntu 22.04 in WSL2 and follow the Linux notes.
Native Windows works for sim but is not actively tested.

### NVIDIA Jetson (JetPack)

Pin numpy and pandas before installing to avoid the system pandas + pip numpy ABI
mismatch:

```bash
pip install "numpy<2" "pandas==2.1.4"
pip install "strands-robots[sim-mujoco,lerobot]"
```

The library detects this case in `dataset_recorder.py` and degrades gracefully if the
ABI is wrong.

## Headless rendering (CI, Docker)

Set `MUJOCO_GL` before importing anything that uses MuJoCo:

```bash
export MUJOCO_GL=osmesa     # software rendering — Linux
# or
export MUJOCO_GL=egl        # hardware rendering with EGL
```

Or in Python before the first import:

```python
import os
os.environ["MUJOCO_GL"] = "osmesa"
from strands_robots import Robot   # safe — imports trigger MuJoCo here
```

## Verifying

```python
import strands_robots
print(strands_robots.__version__)

from strands_robots import list_robots
print(f"{len(list_robots('all'))} robots in the catalog")

from strands_robots import Robot
sim = Robot("so100")
print(sim.tool_name_str)            # 'so100_sim'
sim.step()                          # MuJoCo OK
obs = sim.get_observation("so100")  # rendering OK — numpy arrays per camera
print(list(obs.keys()))             # e.g. ['default']
```

Note: `sim.render()` returns an image content block (PNG bytes in `content`), not a
`["frame"]` key. Use `get_observation(robot_name)` to get raw NumPy arrays, or
`sim.step()` to verify the physics engine runs without rendering.

If any of those fail, see [Troubleshooting](../troubleshooting.md).

## Cache directory

Robot model assets and policy checkpoints are cached under `~/.strands_robots/`:

```
~/.strands_robots/
├── assets/                       # downloaded MJCF + meshes
│   ├── trs_so_arm100/
│   ├── franka_emika_panda/
│   └── ...
└── mesh_audit.jsonl              # safety event audit log (mode 0o600)
```

Override locations with environment variables:

| Env var | What | Default |
|---------|------|---------|
| `STRANDS_ASSETS_DIR` | Robot model assets | `~/.strands_robots/assets/` |
| `STRANDS_MESH_AUDIT_DIR` | Safety audit log | `~/.strands_robots/` |
| `MUJOCO_GL` | GL backend | auto |
| `STRANDS_TRUST_REMOTE_CODE` | Allow HF `trust_remote_code=True` | `false` |
| `STRANDS_ROBOT_MODE` | Default `Robot()` mode | `sim` |
| `STRANDS_MESH` | Disable mesh globally | `true` |
| `GROOT_API_TOKEN` | API token for GR00T service (falls back from `api_token=` kwarg) | unset |

## See also

- [Quickstart](quickstart.md) — five minutes after install.
- [Robot factory](robot-factory.md) — every kwarg `Robot()` accepts.
- [Architecture](../architecture.md) — the optional-extras philosophy.
- [Troubleshooting](../troubleshooting.md) — install gotchas.
