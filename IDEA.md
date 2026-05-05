# IDEA: MJCF AST Refactor — Replace String-Concat Builder with `mujoco.MjSpec`

> **Status:** Proposal / exploration. Nothing implemented. Safe to hand to an autonomous agent as a scoped, staged refactor.
> **Target:** `strands_robots/simulation/mujoco/mjcf_builder.py` + large chunks of `scene_ops.py`
> **Bump required:** `mujoco>=3.2.0` (currently pinned `>=3.0.0,<4.0.0` in `pyproject.toml` — already installed at **3.8.0** in hatch env)

---

## TL;DR

`mujoco.MjSpec` is the official editable MJCF AST shipped with MuJoCo 3.2+. We currently build MJCF by string-concatenating f-strings and mutate scenes by round-tripping XML through `xml.etree.ElementTree`. Switching to `MjSpec` deletes ~600 lines of hand-rolled string munging, kills several known bug classes (camera orientation, keyframe-dim mismatch, mesh-path patching), and unlocks two new capabilities:

1. **Agent-authored raw MJCF** — validated by *actually compiling it*, with a clean fallback path.
2. **Fine-grained live mutation** — add/remove/modify bodies, geoms, sensors, tendons, equalities without a tmpdir + regex roundtrip.

---

## Current state (what's hardcoded today)

### File: `strands_robots/simulation/mujoco/mjcf_builder.py` (273 lines)

Three things baked in:

1. **String-concatenated XML.** Every MJCF element is a Python f-string:
   ```python
   parts.append(f'<geom name="{_sanitize_name(obj.name)}_geom" type="box" '
                f'size="{sx} {sy} {sz}" rgba="{r} {g} {b} {a}" ... />')
   ```
   Every new element type = new f-string. Names require a custom `_sanitize_name` regex to prevent XML injection — which only exists because we're doing string concat.

2. **Shape vocabulary is frozen.** `_object_xml()` is an `elif` ladder over
   `box / sphere / cylinder / capsule / mesh / plane`. `SimObject` has a
   single `shape: str` + `size: list[float]` — no room for `ellipsoid`,
   `hfield`, sites, tendons, equality constraints, pairs, sensors, custom
   materials per-object, friction/solref/solimp tuning, etc.

3. **`_camera_xyaxes_from_target`** — 72 lines of linear algebra + bug-fix
   commentary that exist *solely* because MuJoCo's `mode="fixed"` cameras
   ignore the `target` attribute, so we hand-compute `xyaxes`. With MjSpec
   we just set `cam.targetbody` or let the compiler emit the quat.

### File: `strands_robots/simulation/mujoco/scene_ops.py` (980 lines)

The XML round-trip machinery:

| Helper | Lines | What it does |
|---|---|---|
| `_patch_xml_paths` | ~40 | Rewrites `meshdir`/`texturedir` to absolute paths after `mj_saveLastXML` |
| `_get_abs_meshdir` / `_rewrite_mesh_paths` | ~60 | Patches `<mesh file="...">` paths across robots loaded from different base dirs |
| `_prefix_robot_names` | ~120 | Tree-walks an MJCF root and namespaces every `name=` attribute |
| `_namespace_robot_default_classes` | ~60 | Namespaces `<default class="...">` blocks to avoid collisions on merge |
| `_collect_existing_class_names` | ~15 | Class-name collision avoidance helper |
| `inject_robot_into_scene` | ~50 | Load URDF → save → patch paths → prefix names → merge into scene XML → reload |
| `inject_object_into_scene` | ~34 | `ET.parse` → find `<worldbody>` → append → delete `<keyframe>` (freejoint adds qpos) → write → reload |
| `eject_body_from_scene` | ~45 | `ET.parse` → find body by name → remove → write → reload |
| `eject_robot_from_scene` | ~70 | Same, but also cleans actuators/sensors/equality referencing the robot |
| `inject_camera_into_scene` | ~44 | `ET.parse` → append `<camera>` → reload |

All of this is reimplementing what `MjSpec` gives for free.

### Downstream consumers
- `simulation.py:346` — `xml = MJCFBuilder.build_objects_only(self._world)`
- `simulation.py:292` — `_recompile_world()` rebuilds from scratch via `MJCFBuilder` + `mj.MjModel.from_xml_string`
- `scene_ops.py:790` — `MJCFBuilder._object_xml(obj, indent=4)` called inside `inject_object_into_scene`
- (Search for `MJCFBuilder` to find all call sites.)

---

## Target state: `MjSpec`-backed world

### Core idea
`SimWorld` grows one new field via `_backend_state`:
```python
world._backend_state["spec"]: mujoco.MjSpec
```
`world._model` stays an `MjModel` (unchanged public contract). The *source of truth* for scene structure is the `MjSpec`. The model is derived from it via `spec.compile()`.

### New module: `spec_builder.py` (replaces `mjcf_builder.py`)

Sketch (pseudocode — agent should flesh out):

```python
import mujoco
from mujoco import mjtGeom

SHAPE_MAP = {
    "box":      mjtGeom.mjGEOM_BOX,
    "sphere":   mjtGeom.mjGEOM_SPHERE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "capsule":  mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid":mjtGeom.mjGEOM_ELLIPSOID,  # bonus — free with this refactor
    "mesh":     mjtGeom.mjGEOM_MESH,
    "plane":    mjtGeom.mjGEOM_PLANE,
}

class SpecBuilder:
    @staticmethod
    def build(world: SimWorld) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.angle = mujoco.mjtAngle.mjANGLE_RADIAN
        spec.compiler.autolimits = True
        spec.option.timestep = world.timestep
        spec.option.gravity = world.gravity

        # visual / asset / lights / ground
        SpecBuilder._add_defaults_and_assets(spec, world)

        # cameras — use targetbody or add_frame trick instead of xyaxes math
        for cam in world.cameras.values():
            SpecBuilder._add_camera(spec, cam)

        # objects
        for obj in world.objects.values():
            SpecBuilder._add_object(spec, obj)

        # robots — via attach() (see compose below)
        for robot in world.robots.values():
            SpecBuilder._attach_robot(spec, robot)

        return spec

    @staticmethod
    def _add_object(spec, obj: SimObject):
        body = spec.worldbody.add_body(
            name=obj.name, pos=obj.position, quat=obj.orientation,
        )
        if not obj.is_static:
            body.add_freejoint(name=f"{obj.name}_joint")
            # inertial auto-computed from geoms + mass in MjSpec
        geom_kwargs = dict(
            name=f"{obj.name}_geom",
            type=SHAPE_MAP[obj.shape],
            rgba=obj.color,
        )
        if obj.shape == "mesh":
            geom_kwargs["meshname"] = f"mesh_{obj.name}"
        else:
            geom_kwargs["size"] = _normalize_size(obj.shape, obj.size)
        body.add_geom(**geom_kwargs)
```

No `_sanitize_name` — `MjSpec` validates names itself.
No `_camera_xyaxes_from_target` — use `cam.targetbody` or set `cam.quat` from a helper that MuJoCo's own code verifies.
No f-strings, no escaping, no regex.

### Robot composition (replaces `compose_multi_robot_scene` + all the prefix helpers)

Current ~200 lines of `_prefix_robot_names` + `_namespace_robot_default_classes` collapse to:

```python
robot_spec = mujoco.MjSpec.from_file(robot.urdf_path)  # URDF → spec
frame = scene_spec.worldbody.add_frame(
    pos=robot.position, quat=robot.orientation,
)
scene_spec.attach(robot_spec, prefix=f"{robot.name}_", frame=frame)
```

`attach()` handles:
- Name prefixing across bodies, joints, geoms, actuators, sensors, sites.
- Default class namespacing.
- Asset deduplication (meshes, textures, materials).
- Keyframe merging (or not — configurable).

### Live mutation (replaces `inject_*` / `eject_*`)

`scene_ops.inject_object_into_scene` before:

```python
# tmpdir, save XML, parse with ET, find worldbody, append child,
# delete keyframes, write, reload from path, copy state, re-discover joints
```

After:

```python
def inject_object_into_scene(world, obj):
    spec = world._backend_state["spec"]
    SpecBuilder._add_object(spec, obj)
    world._model, world._data = spec.recompile(world._model, world._data)
    # recompile preserves qpos for unchanged joints; new freejoint qpos = pos/quat from body
    return True
```

`eject_body_from_scene`:

```python
def eject_body_from_scene(world, body_name):
    spec = world._backend_state["spec"]
    body = spec.body(body_name)  # raises KeyError if missing
    body.delete()
    world._model, world._data = spec.recompile(world._model, world._data)
    return True
```

### Agent-authored raw MJCF (the *new* capability)

Add a third tool-facing entry point:

```python
def replace_scene_mjcf(world, xml: str):
    """Atomically swap the whole scene to agent-written MJCF.
    Validated by actually compiling it. Raises on failure
    with the MuJoCo compiler error verbatim.
    """
    new_spec = mujoco.MjSpec.from_string(xml)
    new_model = new_spec.compile()  # raises if invalid
    new_data = mujoco.MjData(new_model)
    world._backend_state["spec"] = new_spec
    world._model, world._data = new_model, new_data

def patch_scene_mjcf(world, ops: list[dict]):
    """Apply a list of structured ops to the live spec.
    ops = [
      {"op": "add_body", "parent": "world", "name": "foo", "pos": [...]},
      {"op": "add_geom", "body": "foo", "type": "box", "size": [...], "rgba": [...]},
      {"op": "set_attr", "path": "body/foo", "attr": "pos", "value": [1,0,0]},
      {"op": "delete",   "path": "body/foo"},
    ]
    """
    spec = world._backend_state["spec"]
    for op in ops:
        _apply_op(spec, op)  # small dispatcher
    world._model, world._data = spec.recompile(world._model, world._data)
```

Both compose cleanly with the `SimObject`/`SimRobot` dataclasses — those remain the *easy path*. Raw MJCF is the *escape hatch*, matching the `_backend_state` pattern already documented in `models.py`.

---

## Work breakdown (staged, safe)

### Stage 0 — Prep
- [ ] Bump `mujoco>=3.2.0` in `pyproject.toml` (`sim-mujoco` optional group). Check current envs; most already have 3.8.
- [ ] Add `strands_robots/simulation/mujoco/spec_builder.py` skeleton. No call sites yet.
- [ ] Unit test: `test_spec_builder_smoke.py` — create `SimWorld` with 2 objects, 1 camera, build spec, compile, assert `model.nbody >= 3`, `model.ncam == 1`, `model.ngeom >= 2`.

### Stage 1 — Parity for object-only scenes (no robots)
- [ ] Implement `SpecBuilder.build(world)` covering everything `MJCFBuilder.build_objects_only` does: visual, asset, lights, ground, cameras, objects.
- [ ] Add feature flag `STRANDS_SIM_USE_MJSPEC=1` in `simulation.py:_recompile_world()` that routes to `SpecBuilder.build(self._world).compile()` vs. the old string path.
- [ ] Ensure hatch env tests pass under *both* code paths.
- [ ] Add a spec-focused test that asserts on spec structure (e.g. `spec.body("cube_1").pos == [...]`), not XML strings.

### Stage 2 — Camera orientation
- [ ] In `SpecBuilder._add_camera`, use `cam.targetbody` when a target is given and a named body at that location exists; otherwise set `cam.quat` from a helper that *uses MuJoCo's own math*.
- [ ] Delete `_camera_xyaxes_from_target` from `mjcf_builder.py` once unused.
- [ ] Port tests in `tests/simulation/test_mujoco_cameras.py` (if they exist — verify).

### Stage 3 — Single-robot attach
- [ ] `SpecBuilder._attach_robot(spec, robot)` using `spec.attach(robot_spec, prefix=..., frame=...)`.
- [ ] Verify joints, actuators, sensors discovered via existing `_discover_*` helpers in `simulation.py` still work (they read from `model`, which is identical downstream).
- [ ] Remove `_save_and_patch_xml` dependency for single-robot scenes.

### Stage 4 — Multi-robot compose
- [ ] Replace `compose_multi_robot_scene` with `SpecBuilder.build(world)` + per-robot `attach()`.
- [ ] Delete `_prefix_robot_names`, `_namespace_robot_default_classes`, `_collect_existing_class_names` once all consumers migrated.
- [ ] Confirm namespace conventions (`{robot_name}_` prefix) match what downstream code reads (grep for joint/actuator name assumptions).

### Stage 5 — Live inject/eject via spec mutation
- [ ] Port `inject_object_into_scene` to `spec.worldbody.add_body(...)` + `spec.recompile(model, data)`.
- [ ] Port `eject_body_from_scene` to `spec.body(name).delete()` + recompile.
- [ ] Port `inject_camera_into_scene`, `eject_robot_from_scene` similarly.
- [ ] Delete `_patch_xml_paths`, `_rewrite_mesh_paths`, `_get_abs_meshdir`, `_save_and_patch_xml` once unused.
- [ ] Handle the `keyframe` qpos-mismatch issue: MjSpec has `spec.keys` — clear or resize appropriately on recompile.

### Stage 6 — Agent-authored raw MJCF
- [ ] Add `replace_scene_mjcf(world, xml)` and `patch_scene_mjcf(world, ops)` in `scene_ops.py`.
- [ ] Expose as Strands `@tool` decorators in `tool_spec.json` + a new tool module. Document clearly: "escape hatch, validated by compilation."
- [ ] Integration test: agent writes a scene with a `<tendon>` element (something `SimObject` can't express), confirms it compiles and simulates.

### Stage 7 — Cleanup
- [ ] Remove feature flag once all stages green in CI.
- [ ] Delete `mjcf_builder.py`.
- [ ] Audit `scene_ops.py` — should shrink from ~980 lines to ~400.
- [ ] Update `AGENTS.md` if the scene-building conventions changed.

---

## Risks & mitigations

1. **`recompile(model, data)` preserves qpos only when joint dims unchanged.**
   *Mitigation:* Adding a freejoint changes `nqpos`. Current code deletes keyframes. Spec version: after recompile, re-inject qpos for unchanged joints by name, leave new joints at their default (body `pos`/`quat`).

2. **`spec.to_xml()` is canonical, not byte-identical to input.**
   *Mitigation:* Any test asserting exact XML strings is wrong and should be rewritten against spec structure or compiled model properties. Grep for `assert.*xml` in tests.

3. **`attach()` default-class naming differs from current `_namespace_robot_default_classes`.**
   *Mitigation:* Concrete difference: `attach(prefix="r1_")` creates `r1_main` default class. Current code may use a different pattern. Find all places that read default-class names (likely none in Python — defaults are consumed by MuJoCo's compiler) and verify. Add integration test with 2 robots from different URDFs to catch regressions.

4. **MuJoCo compiler errors are C-level and sometimes cryptic.**
   *Mitigation:* Wrap `spec.compile()` in `scene_ops.replace_scene_mjcf` with `try/except ValueError` and add context: which spec, which body, what op was being applied.

5. **PR #85 is actively modifying `scene_ops.py`.**
   *Mitigation:* Coordinate with @yinsong1986 before Stage 5. Stages 0–4 are mostly additive and land-safe.

6. **MjSpec API churn 3.2 → 3.8.**
   *Mitigation:* The surface we need (`from_string`, `from_file`, `add_body`, `add_geom`, `attach`, `compile`, `recompile`, `to_xml`, `body.delete`) has been stable since 3.2. Pin `>=3.2.0,<4.0.0` to be safe.

---

## Non-goals

- Not rewriting `physics.py`, `rendering.py`, `randomization.py`, `recording.py` — those consume `MjModel`/`MjData`, which stay unchanged.
- Not changing the Strands tool surface for existing operations — `add_object`, `spawn_robot`, etc. keep their signatures.
- Not changing `SimWorld` / `SimObject` / `SimRobot` / `SimCamera` public fields.
- Not touching Isaac Sim / PyBullet backends (they don't exist yet, but the `SimEngine` ABC is unaffected).

---

## Success criteria

- `mjcf_builder.py` deleted.
- `scene_ops.py` under 500 lines.
- All existing unit + integration tests pass.
- One new integration test proves an agent can author raw MJCF including an element not expressible via `SimObject` (e.g. `<tendon>` or `<equality>`).
- No test asserts on exact XML strings.
- `grep -r "f'<" strands_robots/simulation/mujoco/` returns nothing.

---

## Appendix: proof-of-life snippet

Verified on this host (`mujoco==3.8.0`, 2026-05-05):

```python
import mujoco

# Parse → edit → recompile → serialize
spec = mujoco.MjSpec.from_string('<mujoco><worldbody/></mujoco>')
alice = spec.worldbody.add_body(name='alice', pos=[0, 0, 1])
alice.add_freejoint()
alice.add_geom(name='alice_geom',
               type=mujoco.mjtGeom.mjGEOM_SPHERE,
               size=[0.1, 0, 0], rgba=[1, 0, 0, 1])
model = spec.compile()       # validates; raises on error
assert model.nbody == 2

# Attach a second spec (composition — replaces ~200 lines)
robot = mujoco.MjSpec.from_string(
    '<mujoco><worldbody><body name="arm">'
    '<geom name="link" type="capsule" size="0.05 0.3"/>'
    '</body></worldbody></mujoco>')
frame = spec.worldbody.add_frame(pos=[1, 0, 0])
spec.attach(robot, prefix='r1_', frame=frame)
# Emits: body name="r1_arm", geom name="r1_link", plus a "r1_main" default class.

print(spec.to_xml())  # canonical round-trip
```

---

## Handoff for autonomous agent

When executing this plan:

1. **Work on a feature branch** off `main` — `feat/mjspec-refactor`.
2. **Stage by stage, one PR per stage.** Each PR must be green in CI and reviewable in isolation.
3. **Keep the feature flag alive** until Stage 7. Both code paths tested in CI.
4. **Track progress on the project board** per `AGENTS.md` rule ("the board is the source of truth"): https://github.com/orgs/strands-labs/projects/2
5. **Do not** touch URDF parsing, policy providers, teleoperation, or calibration. Stay inside `simulation/mujoco/`.
6. **Do not** delete anything in `scene_ops.py` until every downstream caller is migrated — audit with `grep -r` before each deletion.
7. **Ask before** bumping any dependency bound other than `mujoco`.
