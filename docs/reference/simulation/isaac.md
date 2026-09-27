# Isaac Sim Backend (GPU)

The Isaac Sim backend runs the simulation on
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) - PhysX physics, RTX
rendering, USD-native scenes. It is a built-in, in-tree backend at
`strands_robots.simulation.isaac`, a peer of `mujoco` and `newton`, implementing
the same `SimEngine` contract, so the `Robot()` / `Simulation` / policy APIs are
identical. Rendering is on the GPU; **physics currently solves on the CPU**
(see [PhysX runs on the CPU](#physx-runs-on-the-cpu)).

`strands-robots` has no hard dependency on Isaac Sim: the `sim-isaac` extra ships
the helpers, and the ~30 GB runtime is provisioned separately.

## When to use it

- You have an NVIDIA RTX GPU (Ubuntu 22.04+, CUDA 12+) and want path-traced
  observations, or RTX metric depth alongside RGB (`get_frame` returns an
  `(H, W) float32` buffer in metres).
- You want USD-native scenes (CAD assets, Nucleus, IsaacLab compatibility).
- You want one scene cloned into many environments - see
  [Fleet replication](isaac-parity.md#fleet-replication). It is not fleet RL: the per-environment
  action/observation API is not implemented.

On macOS, Apple Silicon or a CPU-only host use the MuJoCo backend.

## Install

```bash
# Step 1 - Isaac Sim 6.0 (Python 3.12), via one of:
#   - NGC Docker (the only route verified to RENDER):
#       docker pull nvcr.io/nvidia/isaac-sim:6.0.1
#   - Omniverse Launcher -> Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - pip wheels (physics only - no RTX frames):
#       pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com
#       pip install 'coverage>=7.6.1' && export OMNI_KIT_ACCEPT_EULA=YES

# Step 2 - the sim-isaac extra:
pip install 'strands-robots[sim-isaac]'
```

Use a full `major.minor.patch` docker tag: NVIDIA publishes no `major.minor` tag,
so `:6.0` and `:latest` fail with `no such manifest`. `6.0.1` is the verified tag.

> **The pip route runs physics but produces no RTX pixels**, and nothing raises.
> Measured on an AWS `g5.2xlarge` (A10G), a pip-wheel install steps physics and
> reports success while every RTX camera read comes back empty; the same script
> under `nvcr.io/nvidia/isaac-sim:6.0.1` on that instance returns real frames, and
> it reproduces in pure Isaac Sim. `render()` degrades to a blank frame by
> contract, so a rollout writes an all-black MP4 and reports success, while
> `get_frame()` raises. For pixels, use the container.

`extscache` is required - bare `isaacsim[all]` omits the `isaacsim-extscache-*`
packages and `SimulationApp` aborts resolving its extension graph. Collateral of
the pip install (isaacsim 6.0.0.1, 6.0.1.0):

| Symptom | Cause | Remedy |
|---|---|---|
| An unrelated import dies with `module 'coverage.types' has no attribute 'Tracer'` | `isaacsim-kernel` pins `coverage==7.4.4`, downgrading it under numba | `pip install 'coverage>=7.6.1'`; the reverse pip warning is cosmetic |
| pip conflicts against lerobot's `torchvision` pin | the install upgrades torch, torchvision, numpy, scipy, pyarrow | expected; isaacsim 6.0.x with torch 2.11 / torchvision 0.26.0 and lerobot 0.5.1 is validated |
| `Do you accept the EULA? ... EOF when reading a line` | non-interactive first import | `export OMNI_KIT_ACCEPT_EULA=YES` |
| Exit code 134 after the work completed | a known Isaac Sim atexit segfault | `os._exit(...)` after `SimulationApp` teardown |

Backend discovery is lazy, so MuJoCo-only users never pay the Isaac import cost -
which is why `create_simulation("isaac")` **succeeds** with no Isaac Sim
installed. The failure arrives at `create_world()`, as the structured error every
`SimEngine` method returns, naming each install route:

```python
sim = create_simulation("isaac")     # succeeds - resolves the in-tree backend
sim.create_world()
# {"status": "error", "content": [{"text":
#   "Isaac Sim import failed: omni.isaac.kit.SimulationApp / isaacsim.SimulationApp
#    not available. Isaac Sim must be installed first - via pip ..., Omniverse
#    Launcher, Isaac Lab ..., or Docker (nvcr.io/nvidia/isaac-sim:6.0.1)."}]}
```

The eager check needs no world:

```python
from strands_robots.simulation.isaac import IsaacSimulation
ok, reason = IsaacSimulation.is_available()   # (False, "<every install route>")
```

Newton differs deliberately: it imports its runtime to construct, so
`create_simulation("newton")` raises `ImportError` when `warp` is absent.

## Usage

```python
from strands_robots.simulation import create_simulation

# Kwargs flow into IsaacConfig. "isaac" resolves as a built-in backend.
sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                          # the description MuJoCo loads
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.reset()                                      # see "Adding a dynamic body"
sim.step(120)
frame = sim.render(camera_name="front")          # RGB + depth
sim.destroy()
```

`Robot("so100", backend="isaac", ...)` routes through the same factory. `scale=`
is an accepted alias for `add_object(size=...)` and the only extra keyword that
method reads; any other is refused by name rather than dropped:

```python
sim.add_object(name="cube", heigth=0.3)
# {"status": "error", "content": [{"text":
#   "Unknown parameter(s) ['heigth'] for action 'add_object'. Valid: [...]"}]}
```

## Configuration (`IsaacConfig`)

Keyword arguments to `create_simulation("isaac", ...)` (or
`Robot(..., backend="isaac", ...)`) construct an `IsaacConfig`. Unknown keys are
rejected eagerly, and each field takes the domain its method argument takes.

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `num_envs` | `int` | `1` | Default environment count for `replicate()`, which is what clones them. Setting it alone creates nothing. |
| `device` | `str` | `"cuda:0"` | CUDA device (`cuda:N`) for RTX rendering. PhysX solves on the CPU - see below. |
| `headless` | `bool` | `True` | Run without a GUI (required for cloud/CI). |
| `physics_dt` | `float` | `1/120` | Physics timestep, seconds; positive and finite. |
| `rendering_dt` | `float` | `1/30` | Rendering timestep, seconds. |
| `render_mode` | `str` | `"headless"` | `"headless"`, `"rtx_realtime"` (raster) or `"rtx_pathtracing"` (photoreal). |
| `gravity` | `tuple` | `(0, 0, -9.81)` | Three finite components, Z-up. |
| `ground_plane` | `bool` | `True` | Add a ground plane on `create_world()`. |
| `stage_path` | `str` | `"/World"` | USD prim-path prefix for every created prim: absolute, one component or more, each `[A-Za-z_][A-Za-z0-9_]*`. |
| `nucleus_url` | `str \| None` | `None` | Override the Omniverse Nucleus URL. |
| `camera_width` / `camera_height` | `int` | `640` / `480` | Camera resolution for every `add_camera` / render call stating none of its own. |
| `verbose` | `bool` | `False` | Verbose Isaac Sim / Kit logging. |

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ISAAC_NUCLEUS_URL` | Nucleus server URL, read only when `nucleus_url` is not passed - there the kwarg wins | unset |
| `STRANDS_ISAAC_HEADLESS` | On forces `headless`, off forces windowed; overrides the field | unset |
| `STRANDS_ISAAC_RTX_PATHTRACING` | On forces `render_mode="rtx_pathtracing"`, off leaves it alone; overrides the field | unset |

Which direction the two switches should have is
[#2062](https://github.com/strands-labs/robots/issues/2062). Both take four
symmetric pairs, case-insensitively and ignoring surrounding whitespace: `1`/`0`,
`true`/`false`, `yes`/`no`, `on`/`off`. Unset leaves the field alone, and so does
empty - what an undefined `${{ vars.* }}` interpolation in a GitHub Actions
`env:` block produces. Any other spelling raises `ValueError` naming both
vocabularies rather than falling through to the off side.

### PhysX runs on the CPU

`create_world` builds `World` without passing `device`, and `World`'s own default
is `"cpu"`. `get_state()` reports both - `device` is where physics runs,
`device_requested` is what the config asked for:

```python
sim.get_state()["content"][0]["json"]
# {..., "device": "cpu", "device_requested": "cuda:0", ...}
```

The cost, on an A10G under Isaac Sim 6.0.1 with 40 cuboids and 300 `step()` calls
after a 30-step warmup: **9.9 steps/s on the CPU against 112.3 steps/s** with the
GPU pipeline. It is not enabled because `device="cuda:0"` sets
`gpu_pipeline=True`, and PhysX pre-sizes its tensor buffers at the `world.reset()`
`create_world` performs immediately - for a stage with zero articulations, so the
first `add_robot` fails with `CUDA error: an illegal memory access was
encountered` and the poisoned context takes the next `add_object` with it. The GPU
pipeline needs Isaac Lab's build-then-reset-once pattern, which the incremental
contract here (`create_world`, then `add_robot`, one tool call at a time) does not
have. For physics throughput on a scene known up front, MuJoCo is faster; use this
backend for RTX observations and USD scenes.

### Adding a dynamic body needs a `reset()` before the next `step()`

`add_object` with `is_static=False` (the default) adds a body PhysX's tensor
simulation view does not cover, and that view is what every articulation read goes
through: on an A10G a dynamic cuboid takes a Franka's `get_observation` from 9
joint keys to 0, and removing one makes the next joint read hang. So `step()`
refuses until a `reset()` rebuilds the view:

```python
sim.add_object(name="cube", shape="cuboid", position=[0.4, 0.0, 0.5])
sim.step(60)     # {"status": "error", ...} - the view does not cover the cube
sim.reset()      # rebuilds it
sim.step(60)     # runs
```

`reset()` also returns robots to their default pose, so build the scene and reset
**before** posing anything. A static body needs none of this: `is_static=True`
adds and removes leave the view intact (9 joint keys -> 9), as do `add_camera`,
`move_object`, `add_robot` and `remove_robot`.

## Parity with the other backends

Every `SimEngine` method this backend implements, the contracts specific to it
(description resolution, floating bases, meshes, cameras, the loaders) and
`replicate()` are in [Isaac parity and fleet replication](isaac-parity.md).

## Where to go next

The backend was prototyped in `strands-robots-sim`, whose MkDocs site is kept as
background reference:

- Background docs: <https://strands-labs.github.io/robots-sim/>
- Backend reference: <https://strands-labs.github.io/robots-sim/backends/isaac/>
- Source (historical): <https://github.com/strands-labs/robots-sim>
