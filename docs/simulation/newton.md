# Newton Backend (GPU-native)

The Newton backend runs the simulation on
[newton-physics/newton](https://github.com/newton-physics/newton), NVIDIA's
GPU-accelerated physics engine built on [Warp](https://github.com/NVIDIA/warp)
and MuJoCo-Warp. It implements the same `SimEngine` contract as the MuJoCo
backend, so the `Robot()` / `Simulation` / policy APIs are identical - only the
physics and rendering run on the GPU.

## When to use it

- You have an NVIDIA GPU (Maxwell+, driver 545+, CUDA 12) and want GPU-resident
  physics and rendering.
- You want to choose among Newton's solvers (MuJoCo-Warp, Featherstone, XPBD,
  semi-implicit, ...).
- You want headless rendering without a display server - Newton renders with a
  ray-traced tiled camera sensor, so no GLX/EGL window is required.

On CPU-only hosts Warp falls back to its CPU device; the MuJoCo backend remains
the recommended default for non-GPU machines.

## Install

```bash
uv pip install "strands-robots[sim-newton]"
```

This pulls in `newton`, `warp-lang`, `mujoco-warp`, and `trimesh` on top of the
MuJoCo extra. The backend is lazy-loaded: MuJoCo-only users pay no import cost.

## Usage

```python
from strands_robots.simulation import create_simulation

# "nt" is an alias for "newton".
sim = create_simulation("newton", solver="mujoco")
sim.create_world()
sim.add_robot("so100")          # reuses the same MJCF assets as MuJoCo

sim.send_action({"Rotation": 0.5}, robot_name="so100", n_substeps=100)
print(sim.get_observation("so100"))   # {"Rotation": ..., "Pitch": ..., ...}

# Headless rollout with a recorded video (RGB MP4).
sim.run_policy(
    robot_name="so100",
    policy_provider="mock",
    instruction="wave",
    n_steps=60,
    control_frequency=20.0,
    video={"path": "/tmp/rollout.mp4", "fps": 20, "width": 480, "height": 360},
)
sim.destroy()
```

## Solvers

Pass `solver=` to `create_simulation("newton", solver=...)`. The rigid-body
solvers used by articulated robots are:

| Name | Newton class | Notes |
|------|--------------|-------|
| `mujoco` (default) | `SolverMuJoCo` | MuJoCo-Warp; requires `mujoco-warp` |
| `featherstone` | `SolverFeatherstone` | Reduced-coordinate articulated-body |
| `xpbd` | `SolverXPBD` | Position-based dynamics |
| `semi_implicit` | `SolverSemiImplicit` | Explicit semi-implicit integrator |

`vbd`, `style3d`, `mpm`, and `kamino` are also resolvable for soft-body /
particle scenes but are not exercised by rigid robot arms.

`SolverMuJoCo` requires at least one joint in the model; an empty world (ground
plane only) defers solver creation until a robot is added, and stepping is a
no-op until then.

## Capabilities and parity

- `add_robot` ingests the same MJCF assets as the MuJoCo backend (resolved via
  `strands_robots.assets`). Joint names use the short trailing segment
  (`Rotation`, `Pitch`, ...), matching the MuJoCo backend exactly so policies
  and observation mappings transfer unchanged.
- `render()` returns the same agent-tool image block (`{"image": {"format":
  "png", ...}}`) as MuJoCo, so the shared `PolicyRunner` video pipeline works
  without modification.
- `run_policy` / `eval_policy` / `replay_episode` are inherited from the
  `SimEngine` ABC - no backend-specific re-implementation.
- `describe()` reports the active solver, available solvers, device, and
  the current gravity vector and timestep.
- Gravity configured via `create_world(gravity=[x, y, z])` or `set_gravity`
  drives the dynamics. Newton's solvers snapshot gravity at construction and
  its builder only expresses gravity as a scalar along the up-axis, so the
  full vector is written onto the finalised model before the solver is built;
  off-axis components are honoured rather than dropped. `set_gravity` accepts
  either a scalar (the z-component) or a 3-element `[x, y, z]` list and rebuilds
  the model, which re-initialises the world to its rest pose. `set_timestep`
  takes effect on the next `step()` without a rebuild.

## Limitations

- Only a single default camera view is provided by `render()`; named per-robot
  cameras are not yet supported.
- Mesh objects in `add_object` are not yet supported (primitives box / sphere /
  capsule / cylinder only).
