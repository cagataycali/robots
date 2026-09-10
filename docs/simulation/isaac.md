# Isaac Sim Backend (GPU)

The Isaac Sim backend runs the simulation on
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (PhysX GPU physics +
RTX path-traced rendering). It is a built-in backend at
`strands_robots.simulation.isaac`, a peer of `mujoco` and `newton`, and it
implements the same `SimEngine` contract, so the `Robot()` / `Simulation` /
policy APIs are identical. `strands-robots` has no hard dependency on Isaac Sim:
the `sim-isaac` extra provides the pip-installable helpers, and the ~30 GB
runtime is provisioned separately.

## When to use it

An NVIDIA RTX GPU (Ubuntu 22.04+, CUDA 12+) and one of: photoreal path-traced
observations, USD-native scenes (CAD assets, Nucleus, IsaacLab), Replicator
synthetic data (depth, segmentation, boxes alongside RGB), or fleet RL with
1024+ PhysX environments. On macOS or a CPU-only host use the MuJoCo backend -
the agent contract is identical.

## Install

Install the Isaac Sim runtime first, then the `sim-isaac` extra:

```bash
# Step 1 - install Isaac Sim 6.0 (Python 3.12) via one of:
#   - pip wheels (see caveats below):
#       pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com
#   - Omniverse Launcher -> Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - NGC Docker: docker pull nvcr.io/nvidia/isaac-sim:6.0

# Step 2 - install the sim-isaac extra (helpers for the built-in backend):
pip install 'strands-robots[sim-isaac]'
```

Requesting `create_simulation("isaac")` without the extra raises a `ValueError`
carrying the install hint. Backend discovery is lazy, so MuJoCo-only users
never pay the Isaac Sim import cost.

### Installing Isaac Sim via pip

The cp312 wheels make Isaac Sim 6.0.x pip-installable on Python 3.12. The
`extscache` extra is required, and the install degrades an existing dev
environment in ways pip only warns about, so run this exact sequence:

```bash
# 1. Install the Isaac Sim wheels (NVIDIA index required):
pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com

# 2. Repair the coverage downgrade (see below):
pip install 'coverage>=7.6.1'

# 3. Accept the EULA for non-interactive first import:
export OMNI_KIT_ACCEPT_EULA=YES
```

Known collateral (observed with isaacsim 6.0.0.1 and 6.0.1.0):

| Symptom | Cause | Fix |
|---------|-------|-----|
| An unrelated import dies with `module 'coverage.types' has no attribute 'Tracer'` | `isaacsim-kernel` pins `coverage==7.4.4`, silently downgrading it; numba's tracer probe fails | `pip install 'coverage>=7.6.1'` after the isaacsim install; the reverse pip warning is cosmetic |
| pip conflict warnings against lerobot's `torchvision` pin | isaacsim bumps torch / torchvision / numpy / scipy / pyarrow | Expected; not breakage by itself. Validated 2026-07-31: isaacsim 6.0.x, torch 2.11, torchvision 0.26.0, lerobot 0.5.1. Re-verify your own policy path |
| First import fails with `Do you accept the EULA? ... EOF when reading a line` | Non-interactive first import | `export OMNI_KIT_ACCEPT_EULA=YES` |
| Exit code 134 after successful work | Known atexit segfault in Isaac Sim | Guard scripts that boot `SimulationApp` with `os._exit(...)` after teardown |

## Usage

```python
from strands_robots.simulation import create_simulation

# Kwargs flow into IsaacConfig. "isaac" resolves as a built-in backend.
sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                          # procedural; no asset files needed
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.step(120)
frame = sim.render(camera_name="front")          # RGB + depth
sim.destroy()
```

`Robot("so100", backend="isaac", ...)` routes through the same factory. `scale=`
is an accepted alias for `add_object(size=...)` and the only extra keyword that
method reads; any other keyword is refused by name, as on MuJoCo and Newton:

```python
sim.add_object(name="cube", heigth=0.3)
# {"status": "error", "content": [{"text":
#   "Unknown parameter(s) ['heigth'] for action 'add_object'. Valid: [...]"}]}
```

## Configuration (`IsaacConfig`)

Keyword arguments to `create_simulation("isaac", ...)` (or
`Robot(..., backend="isaac", ...)`) construct an `IsaacConfig`. Unknown keys are
rejected eagerly. The commonly used fields:

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `num_envs` | `int` | `1` | Parallel environments. Set to `1024`+ for fleet RL. A positive integer - the same domain `replicate(num_envs=...)` takes. |
| `device` | `str` | `"cuda:0"` | CUDA device (`cuda:N`). Must be a CUDA device. |
| `headless` | `bool` | `True` | Run without a GUI (required for cloud/CI). |
| `physics_dt` | `float` | `1/120` | Physics timestep (seconds). Positive and finite - the domain `create_world()` applies to the effective dt, and the one the legacy `IsaacSimulation(default_timestep=...)` shortcut that writes this field takes as well. |
| `rendering_dt` | `float` | `1/30` | Rendering timestep (seconds). |
| `render_mode` | `str` | `"headless"` | `"headless"`, `"rtx_realtime"` (raster), or `"rtx_pathtracing"` (photoreal). |
| `gravity` | `tuple` | `(0, 0, -9.81)` | Gravity vector (Z-up). Three finite components, Z-aligned - the same domain `create_world(gravity=...)` takes. |
| `ground_plane` | `bool` | `True` | Add a ground plane on `create_world()`. |
| `stage_path` | `str` | `"/World"` | USD prim-path prefix every created prim is addressed under. Must be absolute, with at least one component, every component a prim name (`[A-Za-z_][A-Za-z0-9_]*`). |
| `nucleus_url` | `str \| None` | `None` | Override Omniverse Nucleus URL (env-resolvable). |
| `camera_width` / `camera_height` | `int` | `640` / `480` | Default camera resolution, for every `add_camera` / render call that states none of its own. Positive integers - the same pixel floor those `width` / `height` arguments take. |
| `enable_rtx_sensors` | `bool` | `True` | Enable RTX-accelerated camera / LiDAR sensors. |
| `verbose` | `bool` | `False` | Verbose Isaac Sim / Kit logging. |

### Environment variables

Three `STRANDS_ISAAC_*` variables are resolved when `IsaacConfig` is
constructed. `STRANDS_ISAAC_NUCLEUS_URL` is read only when `nucleus_url` is not
passed (the kwarg wins); the two switches override their field whenever set -
which direction they *should* have is
[#2062](https://github.com/strands-labs/robots/issues/2062). Both switches are
two-sided and accept four symmetric pairs, case-insensitively, ignoring
surrounding whitespace:

| on | off |
|----|-----|
| `1` | `0` |
| `true` | `false` |
| `yes` | `no` |
| `on` | `off` |

Unset or empty (what an undefined `${{ vars.* }}` in a GitHub Actions `env:`
block produces) leaves the field alone. Any other spelling raises `ValueError`
naming both vocabularies rather than falling through to the off side.

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ISAAC_NUCLEUS_URL` | Override the Omniverse Nucleus server URL when `nucleus_url` is not passed | unset (Isaac defaults) |
| `STRANDS_ISAAC_HEADLESS` | On forces `headless`; off forces windowed | unset (uses `headless` kwarg) |
| `STRANDS_ISAAC_RTX_PATHTRACING` | On forces `render_mode="rtx_pathtracing"`; off leaves `render_mode` alone | unset |

## Capabilities and parity

`IsaacSimulation` exposes the same `SimEngine` shape as the MuJoCo backend:

- **World & lifecycle** - `create_world`, `destroy`, `reset`, `step`,
  `get_state`, `cleanup`.
- **Robots** - `add_robot` (procedural builders, USD via `usd_path=`, or URDF),
  `remove_robot`, `list_robots`, `robot_joint_names`, `send_action`,
  `get_observation`.
- **Objects** - `add_object` (`cuboid` / `sphere` / `cylinder` / `capsule` /
  `mesh`, dynamic or static), `remove_object`. A `shape="mesh"` add takes a
  `mesh_path` (STL/OBJ/MSH, converted to USD once and cached content-addressed
  under `$STRANDS_BASE_DIR/asset_cache/usd_meshes/`, or a USD file);
  the asset defines the extent and collision is its convex hull, the MuJoCo
  contract (Newton reads `size` as a scale,
  [#2300](https://github.com/strands-labs/robots/issues/2300)).
- **Cameras & rendering** - `add_camera` (look-at, FOV), `render` (RGB + depth).
  World-fixed only: `parent_body` is refused with an error naming the backends
  that support it.
- **Loaders** - `load_urdf` / `load_mjcf` / `load_usd` resolve to a
  `ProceduralRobot`. Each XML loader reads its format's own rotation spellings
  and defaults (URDF joint axis +X, MJCF +Z and `hinge`; both free-joint
  spellings), and a URDF joint with no `type` is refused rather than read as
  `fixed`.

The joint-name and observation contract and the accepted *input* domain match
the MuJoCo backend: policies transfer unchanged, and a call one backend refuses
(a malformed pose, `color`, `mass`, camera `fov`, an empty entity `name`) is
refused by all three with the same text. Two Isaac-specific edges: `mass=0` is
refused with `is_static=True` named as the remedy (Newton honours it as that
spelling), and `stage_path` must be an absolute USD prim path of identifier
components, because entity names are interpolated into it.

## Fleet (IsaacLab-style) preview

```python
sim = create_simulation("isaac", num_envs=1024, headless=True,
                        render_mode="headless")
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
# ... RL training loop ...
sim.destroy()
```

## Where to go next

The Isaac backend was originally prototyped in the `strands-robots-sim`
project, which still hosts a MkDocs site with additional architecture notes and
troubleshooting. It is kept here as background reference; the backend itself now
ships in-tree in `strands-robots`:

- Background docs: <https://strands-labs.github.io/robots-sim/>
- Backend reference: <https://strands-labs.github.io/robots-sim/backends/isaac/>
- Source (historical): <https://github.com/strands-labs/robots-sim>
