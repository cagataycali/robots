# Robot Factory

The factory is how `Robot("so100")` knows what to create. It looks up the name in the registry, finds the configuration, and builds the right robot.

---

## How It Works

```python
from strands_robots import Robot

robot = Robot("so100")
```

1. `resolve_name("so100")` checks `registry/robots.json` — finds the canonical name and aliases
2. `_auto_detect_mode("so100")` probes USB ports for Feetech/Dynamixel servos
3. If hardware found → `HardwareRobot` (via LeRobot). If not → simulation backend.
4. Returns the actual backend instance — `MujocoBackend`, `NewtonBackend`, `IsaacSimBackend`, or `HardwareRobot`

---

## Explicit Mode

```python
# Force simulation (even if hardware is connected)
sim = Robot("so100", mode="sim")

# Force real hardware
hw = Robot("so100", mode="real", cameras={
    "wrist": {"type": "opencv", "index_or_path": "/dev/video0"}
})

# Environment variable override
# export STRANDS_ROBOT_MODE=sim
robot = Robot("so100")  # Always sim, regardless of hardware
```

---

## Backend Selection (Sim Only)

```python
# MuJoCo CPU (default) — Mac / Linux / Windows
sim = Robot("so100", backend="mujoco")

# Newton GPU — Linux + NVIDIA
sim = Robot("so100", backend="newton", num_envs=4096)

# Isaac Sim — Linux + NVIDIA
sim = Robot("so100", backend="isaac", num_envs=4096)
```

---

## Name Resolution

The registry resolves aliases. You can use any of these:

```python
Robot("so100")           # Canonical name
Robot("so100_follower")  # Alias
Robot("so_arm100")       # Alias
Robot("trs_so_arm100")   # Alias

Robot("franka")          # Alias → "panda"
Robot("franka_panda")    # Alias → "panda"
Robot("libero_panda")    # Alias → "panda"
```

All aliases are defined in `registry/robots.json` alongside each robot entry.

---

## Return Types

The factory returns the **real backend instance**, not a wrapper:

| Mode | Backend | Returns |
|---|---|---|
| `sim` | `mujoco` | `MujocoBackend` (AgentTool) |
| `sim` | `newton` | `NewtonBackend` |
| `sim` | `isaac` | `IsaacSimBackend` |
| `real` | — | `HardwareRobot` (AgentTool) |

You get full access to all methods of the underlying class.

---

## Zenoh Mesh

By default, every `Robot()` joins a Zenoh peer-to-peer mesh for multi-robot coordination:

```python
robot = Robot("so100", mesh=True)         # Default: mesh enabled
robot = Robot("so100", mesh=False)        # Disable mesh
robot = Robot("so100", peer_id="arm_1")   # Custom peer ID
```
