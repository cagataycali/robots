---
description: A Python environment where Robot imports, MuJoCo renders offscreen and the robot model files are on disk.
---

# Install

At the end of this page you have a Python environment where `from strands_robots import Robot` works, a MuJoCo robot renders offscreen, and the robot model files are on disk.

## Requirements

Python `{{extras:python}}`. A CPU is enough for the MuJoCo path; a GPU is only needed for the Newton and Isaac backends and for policy inference at speed.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "strands-robots[sim-mujoco]"
```

`pip install` works the same way. The bare package has four dependencies (`strands-agents`, `numpy`, `opencv-python-headless`, `Pillow`); everything else is an extra.

## Pick extras

Most people start with `sim-mujoco`: MuJoCo, `robot_descriptions` for asset download, `imageio` for video, `mink` for inverse kinematics. Add `lerobot` when a physical arm arrives, `mesh` when a second machine does.

```bash
uv pip install "strands-robots[sim-mujoco,lerobot]"
```

Every extra, read from `pyproject.toml` at build time:

{{extras:table}}

## Robot models

Simulation loads a robot from its MJCF or URDF. Models are not in the wheel. The first `Robot("so101")` resolves the file through `robot_descriptions` and caches it under `~/.strands_robots/assets/`. Set `STRANDS_ASSETS_DIR` to move that cache.

To fetch models ahead of time, or on a machine that will go offline, call the download tool. It is a plain function as well as an agent tool:

```python
from strands_robots import download_assets

report = download_assets(action="status")
print("\n".join(report["content"][0]["text"].splitlines()[:3]))
```

You should see a count, then one line per robot with a model to fetch, `[ok]` when its files are present and `[--]` when they are not:

```text
<available> available, <missing> missing
[ok] ability_hand         hand         PSYONIC Ability Hand (5-finger prosthetic, 11-DOF)
[ok] adam_lite            humanoid     PNDbotics Adam Lite Humanoid (26-DOF)
```

`action="download"` fetches what is missing; `robots="so101,panda"` narrows the set, `category="arm"` filters by family, `force=True` refetches. A download that fetched nothing returns `status="error"` naming the cause, not a success with zeros.

## Rendering

MuJoCo reads `MUJOCO_GL` once, on first import, and the package sets it when it is unset: on Linux without a display `egl` or `osmesa`, whichever is present; on macOS `cgl`, the only backend, which needs a logged-in window server. Set it yourself only when that choice is wrong:

```bash
export MUJOCO_GL=egl     # Linux, headless, NVIDIA or Mesa EGL
export MUJOCO_GL=osmesa  # Linux, headless, software rendering
```

## Check

```bash
strands-robots doctor
```

No probe touches hardware or the network; [Doctor](doctor.md) lists each and what a `FAIL` means. Then move a robot: [First robot](first-robot.md).
