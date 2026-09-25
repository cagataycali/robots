# Newton Backend (GPU-native)

The Newton backend runs the simulation on
[newton-physics/newton](https://github.com/newton-physics/newton), NVIDIA's
GPU-accelerated physics engine built on [Warp](https://github.com/NVIDIA/warp)
and MuJoCo-Warp. It implements the same `SimEngine` contract as the MuJoCo
backend, so the `Robot()` / `Simulation` / policy APIs are identical - only the
physics and rendering run on the GPU. Scene discovery, the live viewer,
recording, randomization and mesh objects are on
[Newton scenes](newton-scenes.md).

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

It also *narrows* `mujoco`, which `[sim-mujoco]` declares as `>=3.5.0,<4.0.0`,
to the single series the pinned `newton` requires: newton declares that
requirement only under its own `[sim]` extra, so the resolver never applies it
and the pin has to be stated here. `newton` is capped at the next *minor* for
the same reason - the required series is chosen per newton minor - so bumping
newton means moving `mujoco`, `mujoco-warp` and `newton` together.

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

Pass `solver=` to `create_simulation("newton", solver=...)`. The solvers that
integrate a rigid articulated robot are:

| Name | Newton class | Notes |
|------|--------------|-------|
| `mujoco` (default) | `SolverMuJoCo` | MuJoCo-Warp; requires `mujoco-warp` |
| `featherstone` | `SolverFeatherstone` | Reduced-coordinate articulated-body |
| `kamino` | `SolverKamino` | Rigid-body contact solver |

Newton resolves five more names -- `vbd`, `style3d`, `mpm`, `xpbd` and
`semi_implicit` -- that belong to other physics families and have nothing to
integrate in a rigid robot scene. Naming one is refused when the engine is
constructed, with the reason and the list above: unrefused, the first three
raise from inside Newton naming a `ModelBuilder` the caller never touched, and
`xpbd` and `semi_implicit` build and step without moving a joint, so
`add_robot`, `send_action` and `step` all report success over a frozen world.
`describe()["available_solvers"]` reports the accepted names only.

`SolverMuJoCo` requires at least one joint in the model; an empty world (ground
plane only) defers solver creation until a robot is added, and stepping is a
no-op until then.

## Capabilities and parity

- `add_robot` ingests the same MJCF assets as the MuJoCo backend (resolved via
  `strands_robots.assets`) and, because Newton parses URDF natively, also loads
  URDF models directly (see [below](#urdf-robots-via-robot_descriptions)). Joint
  names use the short trailing segment (`Rotation`, `Pitch`, ...), matching the
  MuJoCo backend exactly so policies and observation mappings transfer
  unchanged.
- MJCF **position-servo gains** are carried onto the model. Newton's importer
  reads a `<position>` actuator's `kp` and drops the rest of the servo: the
  `dampratio` MuJoCo compiles into a velocity gain and the `forcerange` that
  caps the torque. A P-only servo with a 1e6 torque ceiling oscillates instead
  of tracking, so both are read off the compiled model and written onto the
  builder before `finalize`. A model MuJoCo cannot compile keeps the gains
  Newton did carry and logs the reason.
- `render()` returns the same agent-tool image block (`{"image": {"format":
  "png", ...}}`) as MuJoCo, so the shared `PolicyRunner` video pipeline works
  without modification.
- `add_camera(name, position, target, fov=60, width, height, parent_body=None)`
  registers named cameras. That order is the same on every backend - Isaac
  included - so a positional call means one thing whichever
  `create_simulation(backend=...)` produced the `sim`. `render(camera_name=...)`
  returns the named view; multiple cameras coexist and `get_observation()`
  returns one RGB frame per camera keyed by name. A `parent_body` (a body label
  from `list_bodies`) mounts the camera ON that body so a wrist camera tracks
  the arm. `remove_camera(name)` / `list_cameras()` round out the API and
  `describe()["cameras"]` lists every registered camera.
- `run_policy` / `eval_policy` / `replay_episode` / `start_policy` are
  inherited from the `SimEngine` ABC - no backend-specific re-implementation. A
  policy needing an action controller only MuJoCo can install is refused there
  rather than rolled out without it: `run_policy` with a `WBCPolicy` reports the
  missing torque shim and names both remedies (the MuJoCo backend, or
  `wbc_install_torque_control=False` for a torque-actuated scene).
  `start_policy` is the ABC's synchronous passthrough to `run_policy` here;
  only the MuJoCo backend runs a policy on a background thread. All four are
  advertised in `describe()["methods"]`, as is every other base-contract
  method this backend delivers.
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

## URDF robots via robot_descriptions

Newton loads URDF natively, so `add_robot` can pull robots straight from the
[`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py)
package - including the large URDF-only long tail (humanoids, quadrupeds, hands,
dual-arm rigs) that has no MJCF model and is therefore unavailable to the MuJoCo
backend.

The `source` selector on `add_robot` controls resolution:

| `source` | Resolution |
|----------|------------|
| `None` (default) | Curated registry / MJCF asset manager first, then a `robot_descriptions` URDF fallback. |
| `"registry"` | Curated registry / MJCF asset manager only (no URDF fallback). |
| `"robot_descriptions"` | URDF directly via `robot_descriptions.<name>_description.URDF_PATH`. |

```python
sim = create_simulation("newton", solver="mujoco")
sim.create_world()

# Load the Franka Panda from its robot_descriptions URDF (no curated entry needed).
sim.add_robot("panda", source="robot_descriptions")
print(sim.robot_joint_names("panda"))
# ['panda_joint1', ..., 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

sim.step(10)   # real GPU model build + step
```

The asset format is detected from the resolved path: `.urdf` files load through
Newton's URDF importer, everything else through the MJCF importer. An explicit
`urdf_path=` argument always wins and bypasses `source`.

`list_urdfs()` returns the union of the registry listing and the
`robot_descriptions` URDF long tail; its `json` block exposes
`robot_descriptions_urdf` (the sorted list of URDF-discoverable names) for
programmatic use. The cheap name lookups behind this
(`urdf_descriptions_module`, `is_urdf_discoverable`, `list_urdf_discoverable`)
live in `strands_robots.registry.discovery` and read a static table with no
import and no network; resolving an actual URDF path clones the upstream asset
repository on first use.

## See also

- [Newton scenes](newton-scenes.md) - discovery and state queries, the live
  viewer, dataset recording, randomization and mesh objects.
- [Isaac Sim backend](isaac.md) and [Isaac parity](isaac-parity.md) - the other
  GPU backend.
