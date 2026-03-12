# Simulation Overview

Run robots in virtual worlds. No hardware needed. Three physics backends, one interface.

<div class="robot-gallery" markdown>
<figure markdown>
  ![SO-100](../assets/so100_act_demo.gif){ width="240" }
  <figcaption><b>SO-100 (ACT policy)</b></figcaption>
</figure>
<figure markdown>
  ![Panda](../assets/panda_act_demo.gif){ width="240" }
  <figcaption><b>Franka Panda (ACT policy)</b></figcaption>
</figure>
<figure markdown>
  ![ALOHA](../assets/aloha_act_demo.gif){ width="240" }
  <figcaption><b>ALOHA Bimanual (ACT policy)</b></figcaption>
</figure>
</div>

---

## Why Simulate?

| Reason | Real World | Simulation |
|---|---|---|
| **Cost** | $500–$50,000 per robot | Free |
| **Speed** | Real-time only | 10–1000× faster |
| **Safety** | Robots break, people get hurt | Nothing breaks |
| **Scale** | One robot at a time | Thousands in parallel |
| **Reproducibility** | Never exactly the same | Deterministic |

---

## Quick Start

```python
from strands_robots import Robot

# Robot() returns a MujocoBackend in sim mode
sim = Robot("so100")  # Auto-detects no hardware → MuJoCo

# Or explicitly:
sim = Robot("so100", mode="sim", backend="mujoco")
```

The factory returns a `MujocoBackend` instance with full tool actions:

```python
# Get observation
obs = sim._dispatch_action("get_observation", {})

# Apply action
sim._dispatch_action("apply_action", {"action": [0.1, -0.2, 0.0, 0.0, 0.0, 0.5]})

# Render a frame
sim._dispatch_action("render", {"path": "frame.png"})
```

---

## MuJoCo Interactive Viewer (mjpython)

MuJoCo provides an interactive 3D viewer window for real-time visualization. On **macOS**, this requires running your script with `mjpython` instead of `python`:

```bash
# Install MuJoCo (included in strands-robots[mujoco])
pip install mujoco

# Run with interactive viewer on macOS
mjpython my_robot_script.py
```

**Why `mjpython`?** macOS requires GUI rendering to happen on the main thread. `mjpython` is a thin wrapper that handles this. On **Linux** and **Windows**, the standard `python` command works for both headless and interactive rendering.

| Platform | Headless `render()` | Interactive Viewer |
|----------|--------------------|--------------------|
| **macOS** | ✅ `python` | ⚠️ Requires `mjpython` |
| **Linux** | ✅ `python` | ✅ `python` |
| **Windows** | ✅ `python` | ✅ `python` |

> **Tip:** For agent-based workflows and CI/CD, headless `render()` is recommended — it returns PNG bytes directly, no window needed.

---

## Three Backends

| | MuJoCo | Newton (GPU) | Isaac Sim |
|---|---|---|---|
| **Physics** | CPU, 1 env | GPU, Warp solver | GPU, 1–100K+ parallel |
| **Platform** | Mac / Linux / Windows | Linux + NVIDIA | Linux + NVIDIA |
| **Install** | `pip install "strands-robots[mujoco]"` | `pip install "strands-robots[newton]"` | `pip install "strands-robots[isaac]"` |
| **Best for** | Development, testing | Fast training | Massive parallel |

```python
# MuJoCo (default)
sim = Robot("so100", backend="mujoco")

# Newton GPU
sim = Robot("so100", backend="newton", num_envs=4096)

# Isaac Sim
sim = Robot("so100", backend="isaac", num_envs=4096)
```

---

## Learn More

- [World Building](world-building.md) — Add objects, change environments
- [Domain Randomization](domain-randomization.md) — Vary physics for robust training
- [Gymnasium Env](gymnasium-env.md) — Standard RL interface
