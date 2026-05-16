# Warp / Newton / MuJoCo-Warp × strands-robots — Integration Research

**Author**: cagatay + DevDuck
**Date**: 2026-05-16
**Status**: Research / Design — pre-RFC
**Source repos analyzed**:
- `https://github.com/NVIDIA/warp` (warp 1.14.0.dev0)
- `https://github.com/newton-physics/newton` (newton 1.3.0.dev0)
- `https://github.com/google-deepmind/mujoco_warp` (MJWarp)
- `git@github.com:strands-labs/robots.git` (this repo)

---

## TL;DR (read this first)

**The landscape changed under us.** A correct understanding:

| What we thought | What's actually true (2026) |
|---|---|
| "Warp has a `warp.sim` module we can use" | **`warp.sim` was REMOVED in Warp 1.x** — superseded by Newton |
| "Warp = simulation framework" | Warp = GPU kernel JIT framework. Newton = physics engine built on Warp. |
| "Warp could replace MuJoCo" | **MuJoCo Warp (MJWarp)** is GPU-accelerated MuJoCo, jointly maintained by Google DeepMind + NVIDIA |
| "It's MuJoCo OR Warp" | Newton uses **MJWarp as its primary backend** — they are complementary, not competitors |

**Three layers, three integration choices:**

```
┌────────────────────────────────────────┐
│ Newton (high-level robotics physics engine)        │  ← New backend candidate
├────────────────────────────────────────┤
│ MuJoCo Warp (MJWarp) (GPU-accelerated MuJoCo)      │  ← Drop-in upgrade for current backend
├────────────────────────────────────────┤
│ NVIDIA Warp (GPU kernel JIT — Python → CUDA)        │  ← Foundation, used directly for custom kernels
└────────────────────────────────────────┘
```

**Recommendation**: ship **two separate backends** under our existing
`SimEngine` ABC, in this order:

1. **`MuJoCoWarpSimEngine`** (Phase 1) — the easy win. Same MuJoCo XML, same
   user-facing API, but GPU-batched. ~10–1000× throughput on parallel scenes.
   Drop-in for users who want speed without API changes.
2. **`NewtonSimEngine`** (Phase 2) — the strategic move. Multi-solver
   (XPBD, Featherstone, VBD, Style3D, MPM, Kamino), differentiable, USD-native,
   future-proof. New abstractions (Newton's `ModelBuilder` is different from
   MuJoCo MJCF), so we keep it as a *peer* backend, not a replacement.
3. **Direct Warp kernels** (opportunistic) — used inside backends for
   specific accelerated ops (raycasting, batch rendering, per-env
   randomization).

Don't pick "one" — ship both as backends, let users choose.

---

## 1. Reality check: what each library is

### 1.1 NVIDIA Warp (the foundation)

- **What**: Python framework that JIT-compiles regular Python functions to
  CUDA / CPU kernels.
- **Status**: 1.14.0.dev0, production-stable, on PyPI as `warp-lang`.
- **Strengths**: differentiable, fast iteration, no CUDA toolkit needed,
  works on CPU + NVIDIA GPU + Apple Silicon (CPU only).
- **What it's NOT**: a physics engine. Not anymore. The historical `warp.sim`
  module **has been removed** — see CHANGELOG: *"Remove `warp.sim` module and
  related examples. This module has been superseded by the Newton library."*
- **What we'd use it for directly in strands-robots**:
  - Custom kernels (e.g. batch domain randomization across N envs)
  - Raycasting / collision queries that aren't in MuJoCo's API
  - Per-env tensor manipulation when running 1000s of parallel sims
  - Tile-based batch rendering (Warp has primitives for this)
- **Risk**: NVIDIA-only for GPU paths. Apple Silicon CPU-only path exists
  but is research-grade.

### 1.2 MuJoCo Warp / MJWarp (the immediate win)

- **What**: GPU-optimized MuJoCo. Same MuJoCo physics, same XML, but the
  step function runs as Warp kernels on GPU.
- **Maintained by**: Google DeepMind + NVIDIA (joint).
- **Status**: on PyPI as `mujoco-warp`, ~3.8.x. Production-track.
- **Compatibility table** (from their README):
  - Dynamics: forward + inverse ✅
  - Transmission, Actuator, Geom, Constraint, Equality: all ✅ (minor exclusions)
  - Integrator: all except IMPLICIT
  - Solver: all except PGS, noslip
  - Sensors: all except PLUGIN
  - Mass matrix: sparse + dense
  - Differentiability via Warp: **not yet**
- **GPU batch renderer included** — ray-traced, multi-camera, mesh+texture+heightfield.
- **Used by**: Isaac Lab (via Newton), MuJoCo Playground (via MJX), mjlab (directly).

**Why this is huge for strands-robots**: our existing `MuJoCoSimEngine`
builds on `mujoco.MjModel` + `mujoco.MjData`. MJWarp exposes a parallel
world API where `nworld` instances of `MjData` step in lockstep on GPU.
**The XML, the spec_builder, the data model are the same.** It's mainly
a new step driver.

### 1.3 Newton (the strategic move)

- **What**: GPU-accelerated physics engine built on Warp, designed for
  robotics simulation. Initiated by Disney Research + Google DeepMind +
  NVIDIA, now a Linux Foundation project.
- **Status**: 1.3.0.dev0, Apache-2.0, on PyPI as `newton`.
- **Solvers**: `xpbd`, `featherstone`, `vbd` (cloth + body), `mujoco`
  (uses MJWarp), `kamino`, `implicit_mpm`, `style3d`, `semi_implicit`.
  **One physics frontend, swappable solvers.**
- **Differentiable**: yes (full forward + reverse mode via Warp).
- **OpenUSD-native**: viewer + recording use USD.
- **Robot input**: URDF, MJCF, USD, custom builders.
- **Examples directory** is rich: `basic/`, `cable/`, `cloth/`, `contacts/`,
  `diffsim/`, `ik/`, `kamino/`, `mpm/`, `multiphysics/`, `robot/`, `selection/`,
  `sensors/`, `softbody/`. Most relevant: `robot/`, `ik/`, `diffsim/`.
- **Caveat**: Newton's `ModelBuilder` is a **different API** from MuJoCo's MJCF.
  We can't just `sim.load_xml("scene.xml")` and have it Just Work — there's a
  conversion layer (Newton ships URDF and MJCF parsers, but the resulting
  `Model` is Newton's, not MuJoCo's).

---

## 2. What strands-robots looks like today

From this repo:

```
strands_robots/simulation/
├── base.py              SimEngine ABC
├── factory.py           create_simulation(), register_backend()
├── models.py            SimWorld, SimRobot, SimObject, SimCamera
├── model_registry.py    URDF/MJCF resolution
├── benchmark.py         BenchmarkProtocol, registry
├── predicates.py
└── mujoco/
    ├── backend.py       lazy mujoco import + GL config
    ├── spec_builder.py  MjSpec-based scene builder
    ├── physics.py       raycasting, jacobians
    ├── scene_ops.py     live recompile via spec
    ├── rendering.py     RGB/depth
    ├── policy_runner.py run_policy / replay / eval_policy
    ├── randomization.py domain randomization
    ├── recording.py     LeRobotDataset recording
    └── simulation.py    Simulation (AgentTool orchestrator)
```

The `__init__.py` of `simulation/` already lists future backends:

```python
# Future backends::
#     from strands_robots.simulation.isaac import IsaacSimulation
#     from strands_robots.simulation.newton import NewtonSimulation
```

**The integration story is already designed.** We just need to fill it in.

---

## 3. Integration architecture

### 3.1 Backend trio under `SimEngine`

```
strands_robots/simulation/
├── base.py                    (existing — SimEngine ABC, unchanged)
├── factory.py                 (existing — add 2 new register_backend calls)
├── mujoco/                    (existing — CPU MuJoCo)
├── mujoco_warp/               ★ NEW backend
│   ├── __init__.py
│   ├── backend.py             lazy mujoco_warp import
│   ├── simulation.py          MuJoCoWarpSimEngine(SimEngine) — GPU-batched MuJoCo
│   ├── batch_rendering.py     wraps MJWarp's GPU batch renderer
│   ├── randomization.py       per-env randomization across nworld
│   └── spec_compat.py         shares MjSpec from existing mujoco/ — zero rewrite
│
├── newton/                    ★ NEW backend (later)
│   ├── __init__.py
│   ├── backend.py             lazy newton + warp imports
│   ├── simulation.py          NewtonSimEngine(SimEngine)
│   ├── model_builder.py       URDF/MJCF → newton.ModelBuilder bridge
│   ├── solvers.py             solver selection helper
│   ├── differentiable.py      diff-sim adapter for policy learning
│   └── usd_recording.py       Newton's USD-based replay
│
└── warp_kernels/              ★ NEW (optional, opportunistic)
    ├── __init__.py
    ├── raycast.py             custom kernels usable from any backend
    ├── randomization.py       parallel domain randomization
    └── batch_obs.py           obs collation across nworld instances
```

### 3.2 SimEngine ABC additions (small, additive)

The ABC in `simulation/base.py` is single-instance-oriented (one world,
one robot, one camera at a time). MJWarp and Newton bring **batched
worlds** — `nworld` parallel scenes.

We add an **optional** `nworld` parameter to `create_world` and a few
optional batched methods, default-implemented to raise `NotImplementedError`:

```python
class SimEngine(ABC):
    # existing methods unchanged...

    @abstractmethod
    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        nworld: int = 1,                                # NEW — default 1 = today's behavior
    ) -> dict[str, Any]: ...

    # New OPTIONAL methods:
    @property
    def supports_batched_worlds(self) -> bool:
        return False

    def get_observation_batched(
        self, robot_name: str, world_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Get observations across N parallel worlds."""
        raise NotImplementedError("Backend does not support batched worlds")

    def send_action_batched(
        self, actions: dict[str, np.ndarray], robot_name: str,
    ) -> dict[str, Any]:
        """actions arrays have leading dim nworld."""
        raise NotImplementedError

    @property
    def supports_differentiation(self) -> bool:
        return False
```

Key property: **existing user code keeps working.** `nworld=1` default
makes the new backends behave identically to the old one for single-env
code.

### 3.3 Factory wiring

```python
# strands_robots/simulation/factory.py
register_backend("mujoco", lambda: MuJoCoSimEngine)        # existing
register_backend("mujoco_warp", lambda: MuJoCoWarpSimEngine)  # NEW
register_backend("newton", lambda: NewtonSimEngine)        # NEW

# Aliases
register_backend("mjwarp", lambda: MuJoCoWarpSimEngine)
register_backend("gpu-mujoco", lambda: MuJoCoWarpSimEngine)
```

User API:

```python
# Today
sim = create_simulation()                    # mujoco, CPU

# Phase 1
sim = create_simulation("mujoco_warp")       # GPU MuJoCo, single env
sim = create_simulation("mujoco_warp", nworld=4096)   # parallel envs

# Phase 2
sim = create_simulation("newton", solver="featherstone")
sim = create_simulation("newton", solver="xpbd", differentiable=True)
```

### 3.4 What we reuse from existing `mujoco/` backend

**Almost all of it.** MJWarp consumes the same MuJoCo XML model produced by
`mujoco.MjSpec`. So:

| Component | Reuse strategy |
|---|---|
| `spec_builder.py` (MjSpec scene builder) | **Reuse as-is** — MJWarp loads MjModel from compiled spec |
| `scene_ops.py` (live recompile) | **Reuse** for single-env, document that batched mode requires reset |
| `rendering.py` | **Add MJWarp batch renderer path**, keep CPU path |
| `policy_runner.py` | Reuse for nworld=1; add batched runner for nworld>1 |
| `randomization.py` | **Replace internals** with MJWarp-native per-env randomization (massive speedup) |
| `recording.py` | Reuse — LeRobotDataset format unchanged. For nworld>1, record one env at a time or stride |
| `physics.py` (raycasting, jacobians) | Some calls need MJWarp equivalents — small porting effort |

**Estimated rewrite**: ~30%. The MJCF authoring layer (the hardest part of
`mujoco/`) is fully reused.

---

## 4. The key wins per backend

### 4.1 MuJoCo Warp (MJWarp) wins

**Throughput** — the headline:
- Single-env CPU MuJoCo: ~1–few kHz on simple scenes
- MJWarp single-env GPU: 2–3× faster for small scenes (overhead dominates)
- **MJWarp nworld=1024**: hundreds of thousands of env-steps/sec
- This makes **massively-parallel RL training** tractable on a single GPU

**Compatibility** — same XML, same spec, same observation shape. Existing
user models load unchanged.

**Built-in batch renderer** — we get GPU ray-traced rendering across all
N cameras and N worlds in a single pass. This **deletes** a lot of
complexity from our `rendering.py`.

**Sim-to-real for VLAs** — the use case that closes the loop with PR #99 +
the Cosmos research:
  1. Spawn 1024 parallel worlds in MJWarp
  2. Run a Cosmos-Predict2.5 robot/policy or GR00T policy across them
  3. Record N×1024 episodes into LeRobotDataset (with Transfer2.5 augmentation
     in the Cosmos research path)
  4. Train on the resulting dataset

### 4.2 Newton wins

**Solver diversity** — each scenario picks its best solver:
- Rigid + articulated bodies: `featherstone` (analytical, accurate)
- Position-based dynamics: `xpbd` (fast, stable, differentiable)
- Cloth / soft contacts: `vbd`
- Cloth-specific (Disney): `style3d`
- MPM (granular materials, fluids): `implicit_mpm`
- Vehicle dynamics: `kamino`
- Drop-in MuJoCo: `mujoco` (same as MJWarp under Newton's API)

**Differentiability** — backprop through physics. Enables:
- Differentiable trajectory optimization
- System identification (learn mass / friction from data)
- End-to-end policy learning with model-based gradients

**OpenUSD** — native scene + recording format. Plays nicely with the wider
NVIDIA Omniverse / Isaac stack and **Cosmos-Transfer2.5** (which understands
USD).

**Future-proof** — Linux Foundation project, multi-vendor (Disney, Google,
NVIDIA). Less likely to die than a single-company project.

### 4.3 Direct Warp kernel wins

**Surgical use, opportunistic.** Examples:

```python
# strands_robots/simulation/warp_kernels/randomization.py
import warp as wp

@wp.kernel
def randomize_friction(rng_states: wp.array[wp.uint32],
                        friction: wp.array2d[float],   # (nworld, nbody)
                        mu_min: float, mu_max: float):
    i, j = wp.tid()
    rs = rng_states[i]
    friction[i, j] = mu_min + (mu_max - mu_min) * wp.randf(rs)
```

This runs in microseconds across millions of (world, body) pairs and is
backend-agnostic — callable from MuJoCo backend OR Newton backend by writing
directly to their underlying buffers.

---

## 5. Dependency strategy

### 5.1 New optional-extras

```toml
[project.optional-dependencies]
sim-mujoco = [                          # existing, unchanged
    "strands-robots[sim]",
    "mujoco>=3.2.0,<4.0.0",
    "imageio>=2.28.0,<3.0.0",
    "imageio-ffmpeg>=0.4.0,<1.0.0",
]

sim-mujoco-warp = [                     # NEW — Phase 1
    "strands-robots[sim-mujoco]",       # MJWarp uses MjSpec from mujoco
    "mujoco-warp>=3.8.0,<4.0.0",
    "warp-lang>=1.13.0,<2.0.0",
]

sim-newton = [                          # NEW — Phase 2
    "strands-robots[sim]",
    "newton>=1.3.0,<2.0.0",
    "warp-lang>=1.13.0,<2.0.0",
    # Newton's own [sim] extra brings mujoco-warp + mujoco for the mujoco solver
]

sim-gpu = [                             # NEW — convenience superset
    "strands-robots[sim-mujoco-warp]",
    "strands-robots[sim-newton]",
]

all = [
    "strands-robots[groot-service]",
    "strands-robots[lerobot]",
    "strands-robots[sim-mujoco]",
    "strands-robots[sim-mujoco-warp]",  # ADD
    "strands-robots[sim-newton]",       # ADD
]
```

### 5.2 Runtime guards

Follow the existing pattern from `mujoco/backend.py`:

```python
# strands_robots/simulation/mujoco_warp/backend.py
import importlib.util as _u

def _ensure_mujoco_warp():
    if _u.find_spec("mujoco_warp") is None:
        raise ImportError(
            "mujoco_warp not installed. "
            "Run: pip install 'strands-robots[sim-mujoco-warp]'"
        )
    if _u.find_spec("warp") is None:
        raise ImportError("warp-lang missing.")
```

No eager imports at module level (matches existing `__init__.py` lazy pattern).

### 5.3 Hardware matrix

| Setup | mujoco | mujoco_warp | newton |
|---|---|---|---|
| Linux + NVIDIA GPU | ✅ | ✅ GPU | ✅ GPU |
| Linux + CPU only | ✅ | ⚠️ Warp CPU mode (slow) | ⚠️ Warp CPU |
| Windows + NVIDIA GPU | ✅ | ✅ | ✅ |
| macOS + Apple Silicon | ✅ | ⚠️ Warp CPU only | ⚠️ Warp CPU only |
| Jetson Thor / Orin | ✅ | **likely ✅** (aarch64 + CUDA, untested by us) | likely ✅ |

Document this prominently. Recommend `mujoco` (existing CPU backend) as
the default. `mujoco_warp` and `newton` are opt-in upgrades.

---

## 6. Concrete user stories

### Story 1: Drop-in throughput upgrade
```python
from strands_robots import create_simulation

# OLD
sim = create_simulation()                           # mujoco CPU

# NEW — same model, same API, ~10× faster on simple scenes
sim = create_simulation("mujoco_warp")
sim.create_world(nworld=1)
sim.add_robot("so100")
# everything else identical
```

### Story 2: Massively-parallel RL
```python
sim = create_simulation("mujoco_warp")
sim.create_world(nworld=4096)
sim.add_robot("so100")

for step in range(10_000):
    obs = sim.get_observation_batched("so100")           # (4096, obs_dim)
    actions = policy.batch_get_actions(obs)              # (4096, act_dim)
    sim.send_action_batched(actions, "so100")
    sim.step()
```

### Story 3: Multi-solver Newton (cloth manipulation)
```python
sim = create_simulation("newton", solver="vbd")
sim.create_world()
sim.add_robot("so100")
sim.add_object("cloth", shape="cloth", material="silk")
# VBD handles soft-rigid coupling natively
```

### Story 4: Differentiable system ID
```python
sim = create_simulation("newton", solver="xpbd", differentiable=True)
sim.create_world()
sim.add_robot("so100")

# Newton exposes Warp tape — backprop through episodes to learn
# mass / friction parameters from real-robot trajectories.
learned_params = sim.diff_optimize(
    target_trajectory=real_robot_recording,
    parameters=["body_mass", "geom_friction"],
    n_iters=200,
)
```

### Story 5: Cosmos + MJWarp + LeRobot end-to-end
*(ties into the Cosmos research doc)*
```python
sim = create_simulation("mujoco_warp")
sim.create_world(nworld=1024)
sim.add_robot("so100")
sim.start_recording(repo_id="my-org/so100-bigdata", world_indices=[0])  # record env 0

policy = create_policy("cosmos_predict", model_id="nvidia/Cosmos-Predict2.5-2B")
sim.run_policy(policy, instruction="pick up red cube", n_episodes=1024)

sim.stop_recording()
# Then offline:
from strands_robots.augmentation import CosmosTransferAugmentor
aug = CosmosTransferAugmentor()
aug.augment_dataset("my-org/so100-bigdata", prompts=["night", "snow", "rain"])
```

---

## 7. Risks and open questions

### 7.1 Risks

1. **API churn**. Newton is `1.3.0.dev0`, MJWarp is `3.8.x` and active.
   Their APIs *will* shift. **Mitigation**: pin tightly (`<2.0`, `<4.0`),
   isolate version-sensitive code in `backend.py`, ship integ tests in CI.
2. **CPU/macOS support is degraded**. Warp's macOS path is CPU-only and
   slow. **Mitigation**: keep `mujoco` (CPU MuJoCo) as the default backend.
   Document `mujoco_warp`/`newton` as Linux+NVIDIA recommended.
3. **Newton's `ModelBuilder` is not MJCF**. We can't perfectly round-trip
   our existing MjSpec scenes. **Mitigation**: Newton ships URDF + MJCF
   parsers; we accept lossy conversion + provide an explicit
   "native Newton scene" path for users who want full Newton features.
4. **Differentiability doesn't compose with all our features**. Live scene
   mutation (recompile), domain randomization mid-episode, etc. break
   gradient flow. **Mitigation**: gate diff-sim behind
   `supports_differentiation` and document what's safe.
5. **GPU memory at high nworld**. 4096 envs with multi-camera renders can
   eat 40GB+. **Mitigation**: render-on-stride pattern, lazy renders, document
   memory model.
6. **Test infrastructure**. Our CI is currently CPU-only. **Mitigation**:
   integ tests gated on `STRANDS_GPU_TEST=1`, run on a GPU runner manually
   or via self-hosted runner. Same pattern as `tests_integ/` for groot.

### 7.2 Open questions

1. **Should MJWarp replace `mujoco` or live alongside it?**
   **Recommendation**: live alongside. CPU MuJoCo is still the right
   default for laptops + macOS + low-N envs. MJWarp wins decisively
   only at nworld≥1.

2. **Should we expose Newton's solvers individually, or just `"newton"`?**
   **Recommendation**: `create_simulation("newton", solver="xpbd")`. The
   solver knob is essential because solver ↔ problem fit is huge
   (XPBD bad for high-stiffness rigid, Featherstone bad for cloth, etc.).

3. **Do we adopt Newton's `ModelBuilder` API, or hide it?**
   **Recommendation**: hide for the basic API (URDF/MJCF in, observation/
   action out, identical to `MuJoCoSimEngine`). Expose via
   `sim.native_builder()` for power users.

4. **Direct Warp kernels in our package: yes or no?**
   **Recommendation**: **yes, but small**. Add `simulation/warp_kernels/`
   with 3–5 high-value kernels (parallel randomization, batch raycast,
   batch obs collation). Don't build a kernel zoo — use Warp opportunistically.

5. **What about Isaac Sim / Isaac Lab?**
   Isaac Lab now uses Newton as a backend (per their `feature/newton` branch).
   So **Newton support implicitly opens an Isaac Lab path**, but we don't
   need to integrate Isaac Lab directly. Defer.

6. **Differentiable simulation is its own product surface.**
   Decide: do we expose `sim.diff_optimize(...)` as a first-class API or
   ship it as an example? **Recommendation**: example first (low commitment),
   API in a Phase 3 if there's demand.

7. **mujoco_warp viewer vs our renderer.**
   `mjwarp-viewer` is a CLI. We have programmatic rendering. **Recommendation**:
   keep ours, optionally surface MJWarp's viewer as a tool for debugging.

---

## 8. Phased plan

### Phase 0 — RFC (this doc)
- [ ] Land `research/WARP_NEWTON_INTEGRATION.md` on `main`
- [ ] Open project-board issue with this body
- [ ] Confirm direction with team

### Phase 1 — `MuJoCoWarpSimEngine` (immediate-win wedge)
**Branch**: `feat/mujoco-warp-backend`

- [ ] New subpackage `simulation/mujoco_warp/`
- [ ] `MuJoCoWarpSimEngine(SimEngine)` — reuses `spec_builder.py` from `mujoco/`
- [ ] Add `nworld: int = 1` to `create_world` (additive, default-preserving)
- [ ] New optional methods `get_observation_batched`, `send_action_batched`
- [ ] Wire batch renderer (replaces N renders with 1 GPU pass)
- [ ] Port `randomization.py` to MJWarp's per-env randomization
- [ ] Update `recording.py` to handle stride/world-index recording
- [ ] Tests: unit (mocked), integ (GPU-gated), benchmark (CPU vs MJWarp)
- [ ] Docs: `docs/simulation/mujoco_warp.md`
- [ ] Add `[sim-mujoco-warp]` extras
- [ ] Register backend in factory

**Acceptance**: `create_simulation("mujoco_warp")` works. Single-env behaves
identically to `mujoco`. `nworld=64` example runs end-to-end. Benchmark
shows expected speedup curve.

**Effort**: 2–3 weeks.

### Phase 2 — `NewtonSimEngine` (strategic move)
**Branch**: `feat/newton-backend`

- [ ] New subpackage `simulation/newton/`
- [ ] `NewtonSimEngine(SimEngine)` — wraps Newton 1.3+
- [ ] URDF / MJCF → `newton.ModelBuilder` bridge (with documented losses)
- [ ] `solver=` parameter + tested combinations
- [ ] USD-based recording option (in addition to LeRobotDataset)
- [ ] Tests + docs
- [ ] Add `[sim-newton]` extras

**Acceptance**: `create_simulation("newton", solver="featherstone")` works
on the same SO-100 URDF as `mujoco`. Solver-specific examples (cloth,
MPM) run.

**Effort**: 4–6 weeks (hardest phase — model translation is non-trivial).

### Phase 3 — Direct Warp kernels (opportunistic)
**Branch**: `feat/warp-kernels`

- [ ] `simulation/warp_kernels/randomization.py` — parallel friction/mass/restitution
- [ ] `simulation/warp_kernels/batch_obs.py` — zero-copy obs collation
- [ ] (Optional) `batch_raycast.py`
- [ ] Used internally by both MJWarp and Newton backends

**Effort**: 1 week.

### Phase 4 — Differentiable simulation example
**Branch**: `examples/diff-sim`

- [ ] `examples/newton_diff_sysid.py` — learn mass/friction from a recording
- [ ] `examples/newton_diff_traj_opt.py` — differentiable trajectory optimization
- [ ] Documented as research-grade

**Effort**: 1–2 weeks.

### Phase 5 — Headline benchmark + demo
**Branch**: `examples/gpu-rl-demo`

- [ ] Benchmark: CPU MuJoCo vs MJWarp at nworld=1, 64, 1024, 4096
- [ ] Demo: train an SO-100 grasping policy on MJWarp at nworld=1024,
  use Cosmos-Predict2.5 for the policy network (closes the loop with
  the Cosmos research)
- [ ] Blog post + chart

---

## 9. What we'd NOT do

1. **Don't drop MuJoCo.** Default backend stays CPU MuJoCo. Reasons: macOS,
   simple debugging, low-N envs, no CUDA dep.
2. **Don't merge MJWarp and Newton into one backend.** They're different
   APIs and have different tradeoffs. Two backends, user picks.
3. **Don't rebuild scene authoring.** `MjSpec` (existing) is the source of
   truth for MuJoCo+MJWarp scenes. Newton uses its own builder — we *bridge*
   from URDF/MJCF, we don't write a new builder.
4. **Don't promise differentiability everywhere.** Only Newton+XPBD/VBD,
   gated by `supports_differentiation` property.
5. **Don't add Isaac Sim as a separate backend yet.** Isaac Lab → Newton is
   the trajectory; we get Isaac compatibility for free via Newton.
6. **Don't fork mujoco_warp / newton / warp.** Treat them as upstream deps.
   Mirrors the rule we set for Cosmos in `COSMOS_INTEGRATION.md`.

---

## 10. Cross-doc relationship

This doc + `COSMOS_INTEGRATION.md` are **complementary**, not competing:

| Concern | Cosmos doc | Warp/Newton doc |
|---|---|---|
| Brain (perception/policy/world model) | covers | — |
| Body (physics + rendering simulation) | — | covers |
| Dataset — *recording* | uses existing recorder | speeds it up via batched MJWarp |
| Dataset — *augmenting* | Transfer2.5 video aug | — |
| Training pipeline | post_train via strands-cosmos | parallel envs via MJWarp |

**Combined headline**: 4096 parallel sims (MJWarp) → Cosmos-Predict2.5 policy
in each → LeRobotDataset recording → Cosmos-Transfer2.5 augmentation →
retrain. **This is the GPU-native robot data flywheel** — and both research
docs are needed to ship it.

---

## 11. Recommendation

**Start with Phase 1 (MJWarp).** Reasons:

- Single biggest user-visible win (throughput) for least effort
- Reuses 70% of existing `mujoco/` backend code
- Validates the optional-extras + GPU-CI story before Newton's bigger surface
- Newton's `mujoco` solver also routes through MJWarp — so this work feeds
  Phase 2 directly
- Doesn't require new ABCs (just additive `nworld=` parameter + 2 optional
  methods)

**Then Phase 2 (Newton)** — once MJWarp is in CI and shaken down.

**Phase 3 (direct kernels)** is opportunistic and small — land it whenever it
unblocks something concrete.

---

## 12. Action items (if approved)

- [ ] Open issue on https://github.com/orgs/strands-labs/projects/2 with
      this doc as body, Status=`Backlog`, Priority=`High`
- [ ] Land this `research/WARP_NEWTON_INTEGRATION.md` on `main`
- [ ] Get +1 on the **two-backend** strategy (MJWarp + Newton, both)
- [ ] Get +1 on **`mujoco` stays default** (no breaking changes)
- [ ] Get +1 on **`nworld=` ABC addition** (single new optional kwarg)
- [ ] Spike `MuJoCoWarpSimEngine` Phase 1 on a feature branch on a GPU box
- [ ] If spike passes: open PR, follow PR-review-learnings checklist from AGENTS.md
- [ ] Coordinate with Cosmos research doc (Phase 5 here = Phase 5 there)
