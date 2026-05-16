# Newton Deep Dive — Can it become *the* strands-robots simulation backend?

**Author**: cagatay + DevDuck
**Date**: 2026-05-16
**Status**: Research / Design
**Source repo analyzed**: `git@github.com:newton-physics/newton.git` (newton 1.3.0.dev0)
**Companion docs**:
- `research/COSMOS_INTEGRATION.md` (the brain)
- `research/WARP_NEWTON_INTEGRATION.md` (overview — read first)
- `research/NEWTON_DEEP_DIVE.md` (this doc — the body, in detail)

---

## TL;DR — the answer to your question

> *"Newton will use mujoco and other sims so we can use the newton as strands-robots implementation, right?"*

**Yes, but with one important nuance.**

Newton CAN be the **single primary backend** for strands-robots. It already:
- Uses **MJWarp** as a built-in solver (`SolverMuJoCo`) — you get GPU MuJoCo for free
- Provides **8 swappable solvers** (XPBD, Featherstone, VBD, Style3D, MPM, Kamino, semi-implicit, mujoco)
- Parses **URDF, MJCF, and USD** natively (`ModelBuilder.add_urdf/add_mjcf/add_usd`)
- Ships **sensors** (IMU, contact, raycast, tiled-camera), **viewers** (GL, USD, Viser, Rerun, file, null), **IK**, **selection** (multi-env articulation views)
- Supports **multi-world batched simulation** natively (no extra abstraction needed)
- Is **differentiable** (Warp tape forward + reverse mode)
- Has **rich robot examples** (G1, ANYmal, Go2, UR10, H1, Allegro, Panda, Cartpole)

**The nuance**: keep the existing CPU `mujoco` backend as a **fallback** for:
1. macOS users (Newton/Warp GPU = NVIDIA only)
2. Quick laptop dev (Newton's compile-on-first-step warmup is ~5–1520s)
3. Single-env workflows where MuJoCo's MjSpec live-recompile (`scene_ops.py`) is
   superior to Newton's `ModelBuilder` (you'd lose live scene mutation).

So the architecture becomes: **Newton = the new default**, CPU MuJoCo = fallback.

---

## 1. What's actually inside Newton (verified by reading source)

### 1.1 Top-level public API

From `newton/__init__.py`:

```python
import newton
import newton.examples
import newton.utils

# Core types
newton.Model, newton.ModelBuilder, newton.State, newton.Control, newton.Contacts
newton.JointType, newton.JointTargetMode, newton.BodyFlags, newton.ShapeFlags

# Submodules
newton.actuators       # PD, PID, clamping
newton.geometry        # Mesh, Heightfield, SDF, Gaussian, TetMesh
newton.ik              # inverse kinematics
newton.math            # quat/vec/transform helpers
newton.selection       # ArticulationView (multi-env)
newton.sensors         # IMU, contact, raytrace, tiled camera
newton.solvers         # 8 solvers
newton.usd             # USD I/O
newton.utils           # download_asset, OnnxRuntime, etc.
newton.viewer          # GL, USD, Viser, Rerun, file, null
```

**Observation**: this is *already* a complete robotics simulation toolkit.
Not a physics primitive — a fully-featured framework. The surface area is
larger than what we currently expose in `strands_robots/simulation/`.

### 1.2 The `Solver` interface

From `newton/_src/solvers/solver.py:301`:

```python
class Solver:
    def __init__(self, model: Model): ...

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Simulate the model for a given time step."""
        raise NotImplementedError()

    def notify_model_changed(self, flags: int) -> None: ...
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None: ...

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None: ...
```

**Concrete subclasses** (in `newton/_src/solvers/`):
- `SolverMuJoCo` — backed by MJWarp + classic MuJoCo CPU. **Most production-ready.**
- `SolverFeatherstone` — analytical articulated dynamics
- `SolverXPBD` — position-based dynamics (fast, differentiable)
- `SolverVBD` — vertex block descent (cloth + body)
- `SolverStyle3D` — Disney's cloth (drape sim)
- `SolverImplicitMPM` — material point method (granular, fluids)
- `SolverKamino` — vehicle-style dynamics
- `SolverSemiImplicit` — simple integrator

All share the same `step(state_in, state_out, control, contacts, dt)` API.
**Solver swapping is one-line.**

### 1.3 The build → finalize → step lifecycle

The canonical Newton loop (from their UR10 example):

```python
import newton
from newton import JointTargetMode
import warp as wp

# Build a single robot scene
robot = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
robot.add_usd("ur10.usda", xform=wp.transform(...))   # or .add_urdf / .add_mjcf

# Replicate to N parallel worlds (built-in!)
builder = newton.ModelBuilder()
builder.replicate(robot, world_count=16, spacing=(2, 2, 0))
builder.add_ground_plane()

# Finalize -> immutable Model
model = builder.finalize()
state_0, state_1 = model.state(), model.state()
control = model.control()
contacts = model.contacts()
solver = newton.solvers.SolverMuJoCo(model)   # one line to swap

# Loop
for _ in range(steps):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0   # ping-pong
```

**This is dramatically simpler than what we have today.** Our current
`MuJoCoSimEngine` carries 9 files (~3000 LOC) of orchestration around
essentially the same loop.

### 1.4 Multi-world (`world_count`) is native

Newton's `ModelBuilder.replicate(sub_builder, world_count=N)` and
`add_world()` mean batched parallelism is **first-class**, not bolted on.
Each entity has a `world` index; solvers respect it; collision detection
respects it; rendering respects it.

From the builder docstring:
> Each entity (particle, body, shape, joint, articulation) has an associated
> world index. Index -1: global entities shared across all worlds. Index
> 0, 1, 2, ...: world-specific entities.

This **deletes** the entire `nworld=` ABC discussion from the Warp/Newton
research doc — we don't need to add a new parameter to `SimEngine`,
because Newton handles it internally.

### 1.5 ArticulationView (the equivalent of our `Robot` selector)

```python
from newton.selection import ArticulationView

view = ArticulationView(model, articulation_id=0)
view.joint_q                  # joint positions
view.joint_qd                 # joint velocities
view.body_q                   # body poses
view.set_joint_target(...)
view.get_dof_count()
```

This maps **directly** to our `SimRobot` dataclass + the `get_observation` /
`send_action` methods on `SimEngine`. We keep our names; the wiring under
the hood points at `ArticulationView`.

### 1.6 Sensors (full suite, GPU-batched)

From `newton/_src/sensors/`:
- `sensor_imu.py` — IMU readings per articulation
- `sensor_contact.py` — contact force/normal per body
- `sensor_frame_transform.py` — frame poses
- `sensor_raycast.py` — GPU batch raycasts (replaces our `physics.py` raycaster)
- `sensor_tiled_camera.py` — GPU tiled camera (replaces most of `rendering.py`)
- `warp_raytrace/` — raytracing primitives

This is **a free upgrade** to our sensor stack. Today our `rendering.py`
renders one camera per call; tiled camera does N cameras across M worlds in
one kernel launch.

### 1.7 Viewers (we get 6 visualization backends)

From `newton/_src/viewer/`:
- `viewer_gl.py` — native OpenGL window (debugging)
- `viewer_usd.py` — record to USD (Omniverse / Pixar viewers)
- `viewer_viser.py` — web viewer (best for remote / SSH workflows)
- `viewer_rerun.py` — [Rerun.io](https://rerun.io) integration (industry-standard logging)
- `viewer_file.py` — file-based recording
- `viewer_null.py` — no-op (CI, headless tests)

**Rerun is huge.** It's becoming the de-facto robotics visualization tool.
Getting it for free shifts our debugging story massively.

### 1.8 IK (inverse kinematics out of the box)

From `newton/_src/ik/`:
- Position + orientation IK
- Multi-target IK
- Examples: Franka, H1, cube stacking, custom

We don't have IK today. Newton gives us a working implementation with
examples. **This is a feature we couldn't easily build in-house.**

### 1.9 Differentiable simulation (built-in)

5 ready-to-run examples in `newton/examples/diffsim/`:
- `example_diffsim_ball.py` — trajectory matching
- `example_diffsim_bear.py` — soft body manipulation
- `example_diffsim_cloth.py` — cloth parameter learning
- `example_diffsim_drone.py` — drone trajectory optimization
- `example_diffsim_soft_body.py` — soft material sysid
- `example_diffsim_spring_cage.py` — stiffness learning

This is functionality we **physically cannot build** without re-implementing
Warp from scratch. It comes free.

### 1.10 ONNX runtime (run RL policies without PyTorch)

From `newton.utils.OnnxRuntime`:
> *"Policies are loaded from ONNX files and run via Newton's Warp-backed
> OnnxRuntime (no PyTorch dependency)."*

This means the *execution* path of a trained RL policy can run **without
PyTorch installed**. Big deal for Jetson / edge deployment.

---

## 2. Mapping Newton onto our existing `SimEngine` ABC

Our ABC (`strands_robots/simulation/base.py`) categories:
- **World lifecycle**: `create_world`, `destroy`, `reset`, `step`, `get_state`
- **Robot management**: `add_robot`
- **Object management**: `add_object`
- **Observation/action**: `get_observation`, `send_action`
- **Rendering**: `render`, `add_camera`
- **Policy orchestration** (concrete in ABC): `run_policy`, `start_policy`, `replay_episode`, `eval_policy`

Direct Newton mapping:

| `SimEngine` method | Newton equivalent | Notes |
|---|---|---|
| `create_world` | `ModelBuilder()` + `add_ground_plane()` | We expose timestep, gravity at finalize time |
| `add_robot(name, urdf_path=...)` | `ModelBuilder.add_urdf(...)` | Plus `add_mjcf`, `add_usd` for free |
| `add_object(name, shape="box", ...)` | `ModelBuilder.add_shape_box/sphere/cylinder/...` | Direct mapping |
| `add_camera` | `newton.sensors.TiledCamera` | GPU-batched! |
| `step(n_steps=N)` | finalize once + N× `solver.step(state_in, state_out, control, contacts, dt)` | We handle ping-pong |
| `reset` | `state.reset()` + restore `joint_q_initial` | Trivial |
| `destroy` | drop refs to model/state/solver, `wp.synchronize()` | Trivial |
| `get_observation(robot_name)` | `ArticulationView(model, id).joint_q/qd` + sensor reads | Multi-world: returns batched arrays |
| `send_action(actions, robot_name)` | `ArticulationView.set_joint_target(...)` | Multi-world: accepts (W, dof) arrays |
| `render(camera_name=)` | TiledCamera read or viewer step | RGB/depth/segmentation |
| `run_policy / replay_episode / eval_policy` | reuse our existing concrete impl | already orchestrates the abstract primitives |

**Verdict**: every method maps cleanly. Newton even has features our ABC
doesn't expose yet (IK, articulation selection, USD recording, Rerun viewer).
Those become **opt-in extensions**.

---

## 3. Architecture proposal

### 3.1 Make Newton the new default

```
strands_robots/simulation/
├── base.py                    (existing — SimEngine ABC, unchanged)
├── factory.py                 (existing — add Newton + change default)
├── mujoco/                    (existing — demoted to fallback backend)
├── newton/                    ★ NEW — the new primary
│   ├── __init__.py
│   ├── backend.py             lazy newton + warp imports
│   ├── simulation.py          NewtonSimEngine(SimEngine)
│   ├── builder_bridge.py      add_robot/add_object → newton.ModelBuilder
│   ├── solvers.py             solver registry + selection
│   ├── articulation.py        ArticulationView wrapper for our Robot model
│   ├── sensors.py             wrap newton.sensors (IMU, contact, raycast, tiled cam)
│   ├── rendering.py           use newton.sensors.TiledCamera + viewer adapters
│   ├── recording.py           LeRobotDataset bridge (reuses dataset_recorder.py)
│   ├── viewers.py             expose newton.viewer choices
│   ├── differentiable.py      gradient-tape adapter (opt-in, Phase 3)
│   └── ik_bridge.py           expose newton.ik (Phase 2)
│
└── (no changes to mujoco/ — it stays as-is, demoted to fallback)
```

### 3.2 Factory default flip

```python
# strands_robots/simulation/factory.py

_DEFAULT = os.getenv("STRANDS_SIM_BACKEND", "newton")   # was "mujoco"

register_backend("newton", lambda: NewtonSimEngine)
register_backend("mujoco", lambda: MuJoCoSimEngine)        # still here, fallback
register_backend("mujoco_warp", lambda: NewtonSimEngine,
                 default_kwargs={"solver": "mujoco"})       # alias
```

User code:

```python
# Today:
sim = create_simulation()                                  # mujoco CPU

# After this PR:
sim = create_simulation()                                  # NEWTON, default solver
sim = create_simulation("newton")                          # explicit
sim = create_simulation("newton", solver="mujoco")         # MJWarp under Newton
sim = create_simulation("newton", solver="xpbd", differentiable=True)
sim = create_simulation("newton", solver="vbd")            # cloth
sim = create_simulation("mujoco")                          # explicit fallback (CPU MuJoCo)
```

### 3.3 Fallback rules (auto-detection)

In `simulation/factory.py:create_simulation()`:

```python
def create_simulation(backend: str | None = None, **kwargs):
    backend = backend or os.getenv("STRANDS_SIM_BACKEND", "auto")

    if backend == "auto":
        # Smart selection
        if _has_nvidia_gpu() and _has_newton():
            backend = "newton"
        elif platform.system() == "Darwin":
            logger.info("macOS detected, falling back to CPU MuJoCo")
            backend = "mujoco"
        else:
            backend = "mujoco"  # safest fallback

    return _registry[backend]()(**kwargs)
```

Documented behavior:
- **Linux + NVIDIA GPU**: defaults to `newton`
- **macOS / Windows-CPU / no-NVIDIA**: defaults to `mujoco`
- Override anytime with `STRANDS_SIM_BACKEND=...` or explicit arg

### 3.4 What we DON'T port forward

`strands_robots/simulation/mujoco/spec_builder.py` (MjSpec live-recompile)
**stays** in the `mujoco/` backend. Newton's `ModelBuilder` requires
finalize-then-step — you can't mutate joints / inertias mid-episode without
recreating the `Model`. If a user needs that workflow, they pick `mujoco`.

This is the **one feature regression** if Newton becomes default. Document
it clearly. Most users don't use live recompile.

---

## 4. The `solver=` design (the key user-facing knob)

### 4.1 Solver ↔ problem-class fit

| Problem | Recommended solver | Why |
|---|---|---|
| Generic rigid manipulation (SO-100, ARM) | `mujoco` (MJWarp) | Most accurate, drop-in MuJoCo XML compat |
| Quadrupeds / humanoids (G1, ANYmal, Go2) | `mujoco` or `featherstone` | High-stiffness rigid contact |
| Cloth / deformables | `vbd` or `style3d` | Built for soft bodies |
| Granular (sand, beans, fluids) | `implicit_mpm` | Material point method |
| Differentiable RL / sysid | `xpbd` | Stable + differentiable |
| Wheeled vehicles | `kamino` | Vehicle-specific |

### 4.2 Smart-default heuristics

```python
def _pick_default_solver(model: newton.Model) -> str:
    # Most users want MJWarp — it's the "familiar" path
    return "mujoco"

# Allow override via env / arg:
sim = create_simulation("newton", solver="xpbd")
# or:
STRANDS_SIM_SOLVER=xpbd python my_script.py
```

### 4.3 Solver switching at runtime

Not supported. You pick once at finalize-time. If you need to switch
solvers, destroy & recreate the simulation. Document this explicitly —
it's the same constraint Newton has.

---

## 5. Differentiable simulation (`differentiable=True`)

### 5.1 Surface

```python
sim = create_simulation("newton", solver="xpbd", differentiable=True)
sim.create_world()
sim.add_robot("so100", urdf_path="...")

# Trainable params (standard PyTorch / Jax / NumPy interop):
import torch
learned_friction = torch.nn.Parameter(torch.tensor(0.5))

for epoch in range(100):
    sim.reset()
    sim.set_geom_friction(learned_friction)        # auto-traced
    for t in range(horizon):
        action = policy(sim.get_observation())
        sim.send_action(action)
        sim.step()
    loss = (sim.get_observation()["position"] - target).pow(2).sum()
    loss.backward()                                # backprop through physics
    learned_friction.data -= 1e-3 * learned_friction.grad
```

### 5.2 Gates

This path is **opt-in** and **research-grade**:
- Only `xpbd`, `vbd`, `featherstone` solvers support full diff. **Not** `mujoco`.
- Live scene mutation breaks gradient flow — reset between episodes.
- Test coverage in `tests_integ/diff/` only.

Gated by `sim.supports_differentiation` property on the engine.

---

## 6. Concrete user stories

### 6.1 Story A: drop-in upgrade
```python
from strands_robots import create_simulation, Robot

# Existing user code works unchanged:
sim = create_simulation()                           # now uses Newton on Linux+GPU
sim.create_world()
sim.add_robot("so100", data_config="so100")
sim.run_policy(my_policy, instruction="pick up cube")
```

### 6.2 Story B: massively-parallel RL
```python
sim = create_simulation("newton", world_count=4096)   # native, no nworld kwarg
sim.add_robot("so100")

view = sim.articulation_view("so100")                 # ArticulationView

for step in range(steps):
    obs = view.get_observation_batched()              # (4096, obs_dim)
    actions = policy.batch(obs)                       # (4096, act_dim)
    view.set_joint_target_batched(actions)
    sim.step()
```

### 6.3 Story C: differentiable system ID (uniquely Newton)
```python
sim = create_simulation("newton", solver="xpbd", differentiable=True)
sim.add_robot("so100", urdf_path="so100.urdf")

# Load real-robot trajectory, learn body masses
learned = sim.diff_optimize(
    target_trajectory=real_recording,
    parameters=["body_mass", "geom_friction"],
    iters=200,
)
```

### 6.4 Story D: cloth manipulation (uniquely Newton)
```python
sim = create_simulation("newton", solver="vbd")
sim.add_robot("so100")
sim.add_cloth("shirt", asset="shirt.usd", initial_state="folded")
sim.run_policy(unfolding_policy, instruction="unfold the shirt")
```
(This is impossible with our current `mujoco` backend.)

### 6.5 Story E: Cosmos + Newton + LeRobot (the headline)
```python
sim = create_simulation("newton", world_count=1024, solver="mujoco")
sim.add_robot("so100")
sim.start_recording(repo_id="my-org/so100-bigdata", world_indices=[0, 100, 500])

policy = create_policy("cosmos_predict", model_id="nvidia/Cosmos-Predict2.5-2B")
sim.run_policy(policy, instruction="pick up red cube", n_episodes_per_world=4)
sim.stop_recording()

# 4096 episodes recorded. Augment with Cosmos-Transfer2.5:
from strands_robots.augmentation import CosmosTransferAugmentor
CosmosTransferAugmentor().augment_dataset(
    "my-org/so100-bigdata",
    prompts=["night", "snow", "rain", "warehouse", "office"],
)
```

### 6.6 Story F: USD export for Omniverse / Cosmos-Transfer2.5
```python
sim = create_simulation("newton", viewer="usd")       # logs to USD
sim.add_robot("so100")
sim.run_policy(policy, instruction="...", n_episodes=10)
sim.export_usd("output.usda")
# Feed output.usda to Cosmos-Transfer2.5 for sim2real video augmentation
```

### 6.7 Story G: Jetson edge inference (no PyTorch)
```python
sim = create_simulation("newton", solver="mujoco")
sim.add_robot("go2", urdf_path="go2.urdf")

# Newton's OnnxRuntime - no torch needed on Jetson
import newton.utils
policy = newton.utils.OnnxRuntime("go2_policy.onnx")

for t in range(steps):
    obs = sim.get_observation("go2")
    action = policy.run(obs)
    sim.send_action(action, "go2")
    sim.step()
```

---

## 7. What we GAIN by adopting Newton

Features we don't have today, free with Newton:

1. **GPU-batched simulation** (1000s of parallel envs)
2. **8 swappable solvers** (cloth, MPM, soft, rigid, articulated, vehicle)
3. **Differentiable physics** (sysid, traj-opt, model-based RL)
4. **Native multi-world model** (no `nworld=` ABC change needed)
5. **TiledCamera** (GPU-batched rendering across worlds)
6. **6 viewers** including **Rerun** and **Viser** (web-native)
7. **OpenUSD I/O** (Omniverse + Cosmos-Transfer2.5 interop)
8. **Built-in IK** (Franka, H1, custom)
9. **Built-in sensors** (IMU, contact, raycast, frame)
10. **ONNX runtime** (PyTorch-free policy execution — huge for Jetson)
11. **Robot examples** (G1, ANYmal, Go2, UR10, H1, Allegro, Panda) we can adapt
12. **ArticulationView selection** (clean API for multi-env joint access)
13. **Linux Foundation governance** (Disney + Google + NVIDIA, won't die)

## 8. What we LOSE / risk

1. **MjSpec live-recompile** (`scene_ops.py`) — can't mutate model post-finalize.
   Workaround: keep `mujoco` backend for users who need this.
2. **macOS GPU path** — Warp on Mac is CPU-only and slow.
   Mitigation: auto-fallback to `mujoco` on Darwin.
3. **NVIDIA-only GPU path** — AMD/Intel users must use CPU.
   Mitigation: same as above.
4. **Pre-1.0 API churn** — Newton is `1.3.0.dev0`.
   Mitigation: pin tightly, isolate version-sensitive code, integ tests in CI.
5. **First-step compile time** — Warp JIT-compiles kernels on first step, can
   take 5–20s for complex scenes.
   Mitigation: documented warmup, kernel cache (Warp supports it).
6. **Larger dependency footprint** — Newton + Warp + mujoco-warp pulls more
   than our current minimal MuJoCo dep.
   Mitigation: gated behind `[sim-newton]` extra. Default `pip install
   strands-robots` doesn't pull Newton.
7. **Docs / examples migration** — our existing examples reference `mujoco`
   backend specifics.
   Mitigation: rewrite examples to use the `create_simulation()` factory
   (already best practice), add migration guide.

**Net assessment**: gains massively outweigh losses. The only hard constraint
is the macOS GPU story — and the auto-fallback solves that.

---

## 9. Does Newton replace `MuJoCoSimEngine` or live alongside?

**Decision: live alongside.** Two reasons:

1. **macOS users / no-GPU users need CPU MuJoCo.** Strands has lots of
   Mac-on-laptop developers. We can't break them.
2. **MjSpec live-recompile is genuinely useful** for some workflows (live
   scene editing in agent loops). It only exists in `mujoco/`.

So:
- `newton` becomes the **default on Linux+NVIDIA**
- `mujoco` stays as **default on macOS / no-GPU** + **opt-in for live editing**
- `MuJoCoWarpSimEngine` from the prior research doc is **deleted from the
  plan** — Newton's `solver="mujoco"` already gives you GPU MuJoCo

**This simplifies the plan from 3 backends to 2.**

---

## 10. Migration path (phased)

### Phase 0 — Land this research doc
- [ ] Commit `research/NEWTON_DEEP_DIVE.md` to `main`
- [ ] Update `research/WARP_NEWTON_INTEGRATION.md` with cross-reference
- [ ] Project-board issue

### Phase 1 — `NewtonSimEngine` skeleton (GPU rigid only)
**Branch**: `feat/newton-backend-skeleton`

Minimum viable backend: rigid body only, solver="mujoco", single + multi-world.

- [ ] `simulation/newton/` package skeleton
- [ ] `NewtonSimEngine(SimEngine)` implementing required ABC methods
- [ ] `add_robot` via `ModelBuilder.add_urdf` / `add_mjcf` (USD later)
- [ ] `add_object` via `add_shape_*` family
- [ ] `step` / `reset` / `destroy`
- [ ] `get_observation` / `send_action` (single + batched)
- [ ] Render via `newton.sensors.TiledCamera`
- [ ] Recording bridge (reuse `dataset_recorder.py`)
- [ ] `[sim-newton]` extras
- [ ] Unit tests (mocked) + integ tests (GPU-gated)
- [ ] Side-by-side benchmark vs `mujoco` backend on SO-100 grasp

**Acceptance**: `create_simulation("newton")` runs SO-100 + cube grasp,
produces same observation shapes as `mujoco`, `world_count=64` works.

**Effort**: 3 weeks.

### Phase 2 — Solver registry + sensor wrappers
**Branch**: `feat/newton-solvers-sensors`

- [ ] `solver="xpbd|featherstone|vbd|mujoco|..."` parameter
- [ ] Smart defaults per scene type
- [ ] Wrap `newton.sensors.IMU` / `Contact` / `Raycast` / `TiledCamera`
- [ ] Expose IK via `sim.ik(...)`
- [ ] USD export: `sim.export_usd(path)`
- [ ] Rerun viewer integration (huge debugging win)
- [ ] Per-solver examples

**Effort**: 2–3 weeks.

### Phase 3 — Differentiable simulation
**Branch**: `feat/newton-diffsim`

- [ ] `differentiable=True` mode
- [ ] `sim.diff_optimize(...)` API
- [ ] PyTorch + JAX interop tests
- [ ] System ID example (mass / friction from real recording)
- [ ] Trajectory optimization example
- [ ] Document gradient-safety constraints

**Effort**: 2 weeks.

### Phase 4 — Make Newton the default
**Branch**: `feat/newton-default`

- [ ] Auto-backend selection in `create_simulation()`
- [ ] `STRANDS_SIM_BACKEND` env var
- [ ] Migration guide doc
- [ ] Update all `examples/*.py` to be backend-agnostic
- [ ] Update README headline ("Now powered by NVIDIA Newton...")
- [ ] Deprecation warning if user passes a `mujoco`-only kwarg without
      explicitly choosing the `mujoco` backend

**Effort**: 1 week.

### Phase 5 — Integration with Cosmos research
**Branch**: `feat/newton-cosmos-loop`

- [ ] Headline demo: 1024 envs + Cosmos-Predict2.5 policy + LeRobotDataset
- [ ] USD pipeline to Cosmos-Transfer2.5
- [ ] Benchmark / blog post

**Effort**: 2 weeks.

---

## 11. Comparison matrix (decision aid)

| Concern | Keep `mujoco` (today) | Add `mujoco_warp` only | **Adopt Newton** |
|---|---|---|---|
| GPU-batched | ❌ | ✅ | ✅ |
| Solver diversity | ❌ (MuJoCo only) | ❌ (MuJoCo only) | ✅ (8 solvers) |
| Differentiable | ❌ | ❌ | ✅ |
| Native multi-world | ❌ (we'd add `nworld=`) | ❌ (we'd add `nworld=`) | ✅ (built-in) |
| Rendering (GPU batched) | ❌ | ✅ (MJWarp renderer) | ✅ (TiledCamera) |
| Sensors (IMU, contact, raycast) | partial | partial | ✅ (full suite) |
| Viewers (Rerun, Viser, USD) | ❌ | ❌ | ✅ |
| IK (built-in) | ❌ | ❌ | ✅ |
| ONNX runtime (no torch) | ❌ | ❌ | ✅ |
| OpenUSD I/O | ❌ | ❌ | ✅ |
| URDF/MJCF/USD parsers | MJCF only | MJCF only | ✅ (all three) |
| Future-proof governance | mujoco upstream | mujoco_warp + warp | Linux Foundation |
| macOS support | ✅ | ⚠️ CPU only | ⚠️ CPU only (fallback to mujoco) |
| Effort (LOC) | 0 | ~1500 | ~2500 |
| Surface area gained | — | — | enormous |

**Recommendation**: **adopt Newton**. The work is bigger but the gain is
disproportionate — we don't have to write IK, multi-world support, sensors,
USD I/O, Rerun integration, etc. We get all of it.

---

## 12. Cross-doc reconciliation

This doc **supersedes** the dual-backend plan in `WARP_NEWTON_INTEGRATION.md`:

| Decision in WARP_NEWTON doc | Updated decision (this doc) |
|---|---|
| Add `MuJoCoWarpSimEngine` as separate backend | **Drop**. Use `Newton(solver="mujoco")` instead |
| Add `NewtonSimEngine` later | **Promote**. Make it the new default |
| Add `nworld=` parameter to `SimEngine` ABC | **Drop**. Newton handles it natively |
| Add `simulation/warp_kernels/` | **Optional**. Only if specific kernels are needed |
| Direct Warp kernel usage in our code | **Discouraged**. Newton encapsulates this |

The revised plan: **2 backends**, not 3.
- `newton` (new default, GPU-batched, multi-solver, diff-sim, MJWarp-inside)
- `mujoco` (CPU fallback for macOS / live-recompile workflows)

---

## 13. Action items

- [ ] Cross-link this doc + supersede `WARP_NEWTON_INTEGRATION.md` Phase 1
- [ ] File issue on https://github.com/orgs/strands-labs/projects/2,
      Status=`Backlog`, Priority=`High`, link both Cosmos + Newton docs
- [ ] Get +1 from team on "Newton becomes default backend"
- [ ] Get +1 on "keep `mujoco` as fallback (macOS, live-recompile)"
- [ ] Get +1 on "drop `MuJoCoWarpSimEngine` plan in favour of `Newton(solver='mujoco')`"
- [ ] Spike Phase 1 (`NewtonSimEngine` skeleton) on a GPU box
- [ ] Update `cagataycali/strands-gtc-nvidia` autonomous task issue with this
      revised architecture
- [ ] Coordinate with Cosmos research — Phase 5 demo combines both
