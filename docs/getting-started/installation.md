---
description: pip install strands-robots — extras matrix, platform notes, headless rendering.
---

# Installation

`strands-robots` is a single `pip install`. Pick the extras that match your use case.

## Extras matrix

| Extra | Pulls in | When you need it |
|-------|----------|------------------|
| (none) | core only — Robot factory, registry, lazy imports | Inspect the catalog, write tools |
| `[sim-mujoco]` | `mujoco`, `numpy`, `imageio` | Any `Robot()` with default `mode="sim"` |
| `[lerobot]` | `lerobot`, `torch` | Real hardware OR `LerobotLocalPolicy` |
| `[groot-service]` | `pyzmq`, `msgpack` | `Gr00tPolicy` (talks to a GR00T container) |
| `[mesh]` | `eclipse-zenoh` | Multi-robot mesh discovery + RPC |
| `[benchmark-libero]` | LIBERO eval deps | LIBERO benchmark suite |
| `[all]` | everything above | Demos, CI, exploration |
| `[dev]` | `pytest`, `ruff`, `mypy`, `hatch` | Contributing |

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

LeRobot + torch. Skip `[sim-mujoco]` if you'll never simulate.

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
print(sim.render()["frame"].shape)  # rendering OK
```

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

## See also

- [Quickstart](quickstart.md) — five minutes after install.
- [Robot factory](robot-factory.md) — every kwarg `Robot()` accepts.
- [Architecture](../architecture.md) — the optional-extras philosophy.
- [Troubleshooting](../troubleshooting.md) — install gotchas.
