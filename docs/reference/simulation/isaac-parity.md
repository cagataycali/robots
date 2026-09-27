# Isaac parity and fleet replication

What the [Isaac Sim backend](isaac.md) implements of the `SimEngine` contract,
where it answers differently from MuJoCo and Newton, and what `replicate()`
builds. The guide page covers install, configuration and the two runtime
caveats.

## Capabilities and parity

`IsaacSimulation` exposes the same `SimEngine` shape as the MuJoCo backend: world
and lifecycle (`create_world`, `destroy`, `reset`, `step`, `get_state`,
`cleanup`), robots (`add_robot`, `remove_robot`, `list_robots`,
`robot_joint_names`, `send_action`, `get_observation`), objects (`add_object`,
`remove_object`) and cameras (`add_camera`, `render`). The backend-specific
contracts:

| Area | Contract here |
|---|---|
| `add_robot` source | An explicit `usd_path=`, `urdf_path=` or `mjcf_path=`; with none, the name (or `data_config=`) resolves through the **same** registry resolver MuJoCo uses, so one name loads one description. An MJCF is converted to USD once via `isaacsim.asset.importer.mjcf` and cached under `$STRANDS_BASE_DIR/asset_cache/usd_robots/`, content-addressed over the description *and every file in its directory* (an `<include>` and a `meshdir` are part of it). An unresolvable name gets MuJoCo's three-way diagnosis: a typo with close matches, a hardware-only entry, or an asset not downloaded. |
| Floating base | `add_robot(..., fix_base=False)` gives a URDF robot a floating base, as a humanoid or quadruped needs; the default `True` welds the root. It is a parameter because URDF cannot answer it - a mobile root link is byte-identical to a bolted-down arm's base - while MJCF states it with `<freejoint>`, so MuJoCo and Newton have no equivalent. On a USD asset, which carries its own articulation root, `fix_base=False` is refused rather than ignored. A floating base reports the four `base_*` entries the observation schema requires (`base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`) and a fixed-base arm reports none; `reset()` does not preserve its spawn height - see `add_robot`'s docstring. |
| Meshes | `shape="mesh"` takes a `mesh_path` to an STL/OBJ/MSH asset (converted once, cached under `$STRANDS_BASE_DIR/asset_cache/usd_meshes/`) or to a USD file. The asset defines the extent, so `size` is ignored - MuJoCo's read of it, where Newton takes a scale ([#2300](https://github.com/strands-labs/robots/issues/2300)) - and collision uses the **convex hull**, which fills a concave cavity. A missing file, an unconvertible format or a non-finite vertex is refused up front, never realized as a fallback primitive. |
| Cameras | World-fixed only. `parent_body` (a wrist camera on mujoco/newton) is refused with an error naming those backends: camera prims parent to the stage camera scope, not to an articulation link. |
| Loaders | `load_urdf` / `load_mjcf` / `load_usd` return a `ProceduralRobot`: the description-**introspection** API for tooling and the parity tests, not the load path. Each format's own defaults apply, so a joint is never read under the other's - an MJCF `<joint>` with no `type` is a hinge about +Z, a URDF joint must state `type` and acts about +X - and both MJCF spellings of a free joint are reported with `joint_type="fixed"`, visible in `joints` without counting as an actuated DOF. |
| Scenes | `load_scene` renders each object with its real mesh while keeping the validated collision-AABB box as an invisible physics proxy, so switching backends does not switch what the cameras see. An object whose mesh cannot be resolved keeps a visible box proxy, and the report then carries an explicit caveat that pixel-conditioned scores on that scene are not comparable across backends. |

Because the joint-name and observation contract matches MuJoCo, policies and
observation mappings transfer unchanged: for a robot named rather than pathed both
backends read the names from one asset. Measured on Isaac Sim 6.0.1, `panda`
reports `joint1`..`joint7` plus `finger_joint1`/`finger_joint2` (9 of 9 matching
MuJoCo) and `so100` reports `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`,
`Wrist_Roll`, `Jaw` (6 of 6). An explicit `usd_path=` opts out of that guarantee.

The accepted input domain matches too - the pose vectors, an object's `color` and
`mass`, the camera `fov` and pixel dimensions - so a call one backend refuses is
refused by all three. Where this backend answers differently:

| Input | Verdict here |
|---|---|
| entity `name` | A non-empty `str` with no NUL, on `add_robot`, `add_object` and `add_camera`. It is interpolated into the prim path (`{stage_path}/Robots/{name}`), where `""` addresses the container scope every robot shares and `remove_robot` prunes its registry by that prefix. There is no derive-a-label short form: `name` is also the key registry resolution falls back to without `data_config=`. |
| `mass=0` | Refused, with `is_static=True` named as the remedy. Newton documents it as a spelling of `is_static=True` and honours it. |
| an unknown entity, on a lookup | Answered, not raised: `remove_robot`, `remove_object`, `remove_camera`, `send_action`, `move_object` and `get_body_state` report the unknown-entity message, `robot_joint_names` and `get_observation` answer empty, and `get_frame` / `get_camera_params` raise the `KeyError` their contract names. |
| `render(camera_name=...)` naming an absent camera | `{"status": "error"}` with `Camera '<name>' not found. Available: [...]` - the message `get_frame` raises and the one MuJoCo and Newton give. |
| `render` with a name that names *no* camera (`None`, `""`, `"default"`, `"free"`) | A blank frame. Isaac has no free camera to fall back to, so for it that is a degradation rather than a mistake; registering a camera under one of those names is accepted and renders normally. |

## Fleet replication

`replicate()` clones the scene you have built into a grid of parallel environments
with Isaac Sim's own `isaacsim.core.cloner.GridCloner`. The scene on the stage is
environment 0, so `num_envs` counts it - `replicate(64)` produces the source plus
63 clones under `{stage_path}/envs/env_1 .. env_63`:

```python
sim = create_simulation("isaac", num_envs=64, headless=True,
                        render_mode="headless")
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")

result = sim.replicate(64, spacing=1.5)   # or replicate() to use config.num_envs
# {"status": "success", ... "Cloned the scene into 64 environments (63 clones of
#  1 source prim(s) plus the source as env_0, 1197 prims) in 1840ms at 1.50m
#  spacing on cuda:0. NOTE: get_observation/send_action address env_0 only ..."}

sim.destroy()
```

Physics is replicated and inter-environment collisions are filtered, so the clones
do not push each other around. The payload reports what was built -
`clones_created`, `prims_created`, `build_time_ms`, `physics_replicated`,
`collisions_filtered` - rather than the count you passed.

**What is not implemented: a per-environment observation or action API.**
`get_observation` and `send_action` address environment 0's robot, the only one
carrying an `Articulation` handle. The clones advance under physics and are what a
renderer and a domain-randomisation pass see, but they cannot be driven or read
individually; every success message says so, so a `num_envs: 64` in the payload is
not mistaken for 64 drivable robots.

`replicate(1)` is an accepted no-op that deliberately does *not* mark the
simulation replicated, so `add_robot` keeps working; `replicate(n > 1)` marks it
and `add_robot` is refused from then on, since the clones were built from the
scene as it stood. `num_envs` and `spacing` take the shared numeric domains, so a
negative count or a NaN spacing is refused rather than reported as a fleet.

