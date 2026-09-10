---
description: The Simulation AgentTool - every action grouped by category, with parameters.
---

# Simulation overview

```python
from strands_robots import Robot
sim = Robot("so100")   # preferred factory; 60+ actions as an AgentTool
```

Every action below is also listed in `sim.describe()["methods"]`, so an agent
can discover the whole surface from one call. Numeric parameters share one
domain everywhere: a value is refused with a structured `status="error"` when
it is a boolean (`float(True)` is `1.0`, and `numpy.bool_` is refused too),
`nan` / `inf`, or a vector of the wrong component count; a refused call
changes nothing. For composing scenes see [World building](world-building.md).

## World

| Action | Key params | Notes |
|--------|-----------|-------|
| `create_world` | `timestep=0.002`, `gravity=[0,0,-9.81]`, `ground_plane=True` | Implicit on `Robot()` |
| `load_scene` | `scene_path` | Replace world with MJCF |
| `reset` | - | State to t=0, keep model |
| `get_state` | - | Sim time, joint positions, object poses |
| `destroy` | - | Tear down model, data, executor |
| `export_xml` | `output_path` | Serialise live scene to MJCF; reloadable via `load_scene` (assets referenced by absolute path) |

## Scene-MJCF

| Action | Notes |
|--------|-------|
| `replace_scene_mjcf(xml)` | Swap entire world XML |
| `patch_scene_mjcf(ops)` | Incremental patches, no full recompile |
| `raycast(origin, direction, ...)` | Single ray–mesh intersection |
| `multi_raycast(origin, directions, ...)` | Batch ray–mesh intersections from one origin; all-or-nothing, a direction it cannot cast refuses the batch |

## Robots

| Action | Key params |
|--------|-----------|
| `add_robot` | `robot_name`, `position=[0,0,0]`, `data_config=None`, `urdf_path=None` |
| `remove_robot` | `name` |
| `list_robots` | - |
| `get_robot_state` | `name` → joint positions, velocities, torques |

## Objects

| Action | Key params |
|--------|-----------|
| `add_object` | `name`, `shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh"`, `size`, `position=[x,y,z]`, `color=[r,g,b,a]`, `orientation=[w,x,y,z]`, `mass=0.1`, `is_static=None`, `mesh_path=None` - omitted lets the shape decide: `plane` is made static and refuses an explicit `is_static=False`, every other shape is dynamic |
| `remove_object` | `name` |
| `move_object` | `name`, `position`, `orientation` (NOT `pos`/`quat`) |
| `list_objects` | - |

## Cameras

| Action | Key params |
|--------|-----------|
| `add_camera` | `name`, `position`, `target`, `fov=60.0`, `width=640`, `height=480` - no `attach_to`/`fovy`/`lookat` |
| `remove_camera` | `name` |
| `list_cameras` | - renderable camera names, `"default"` first, incl. model + user cameras |

Robot-URDF cameras are auto-discovered on `add_robot`. `list_cameras()` returns
every name `render` / `start_recording` accepts - `"default"` first, then
model-defined and `add_camera` cameras - and equals `sim.describe()["cameras"]`.

## Rendering

| Action | Notes |
|--------|-------|
| `render(camera_name="default", width=None, height=None)` | PNG in `content[...]["image"]["source"]["bytes"]`; no `frame` key |
| `render_depth(camera_name="default", width=None, height=None)` | Viewable grayscale depth PNG `image` block (near=bright, far=dark) + metric `depth_min`/`depth_max` (meters) in the `json` block |
| `render_all(cameras=None, width=None, height=None)` | One `image` block per camera (multi-view snapshot) |
| `get_world_point(camera_name="default", pixels=[[u, v], ...])` | Ground picked pixels to metric world coordinates via the depth buffer; `point` is the median over the valid samples, `points` aligns with the input pixels |
| `open_viewer` / `close_viewer` | Interactive MuJoCo passive viewer |

`sim.get_observation(robot_name)[camera_name]` returns the frame as
`np.uint8 (H, W, 3)`. Frame reads copy `mjData` under the simulation lock, so a
frame captured while a policy or recorder is stepping is a consistent snapshot.
`sim.get_camera_params(camera_name)` returns the pinhole `K` the renderer
actually draws with - a camera declaring an MJCF sensor (`sensorsize` /
`focal` / `principal` / `resolution`) keeps its non-square pixels and
off-center principal point; every other camera derives `K` from `fovy`.

## Physics

| Action | Key params |
|--------|-----------|
| `step` | `n_steps=1` (MuJoCo: max 100 000/call; Isaac and Newton have no ceiling). Non-negative whole number; `0` is an accepted no-op. Errors if the world is destroyed mid-run, naming the steps completed |
| `send_action` | `n_substeps=1` - **positive** whole number, no per-call ceiling (see Actions) |
| `set_gravity` | `gravity=[x,y,z]` or a scalar z-component |
| `set_timestep` | `timestep` |
| `get_contacts` / `get_contact_forces` | - . `get_contacts` lists every geom pair inside the detection range (`margin` + `gap`) and marks each one `active` - MuJoCo hands only the pairs inside `margin` to the solver, so a pair between the two thresholds is a proximity report carrying no force. Contact predicates count only `active` pairs; `get_contact_forces` gives the load a touching pair carries |
| `apply_force` | `body_name`, `force`, `torque`, `point` - latched on that body and re-applied every step until the next `apply_force` for it, so several bodies can hold wrenches at once (`force=[0,0,0]` stops one, `reset()` stops all) |
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name`. Columns are DOFs of the whole compiled model, so the width is not the robot's joint count: a free or ball joint owns several consecutive columns, and a scene holding two robots reports one width spanning both. The `json` block's `dof_joint_names` names the joint owning each column - pair `dq` with that, not with `robot_joint_names`, or one robot's Jacobian reads as another's |
| `get_mass_matrix` | - . The reported `diagonal` is DOF-indexed on the same terms, and `dof_joint_names` names each entry's joint |
| `inverse_dynamics` | - (compensation torques to hold the current `qpos`/`qvel`) |
| `forward_kinematics` | `body_name` (optional) |
| `save_state` / `load_state` | `name` - snapshot/restore full physics. A checkpoint is valid only for the model it was taken against: any scene mutation that swaps the compiled model (`add_object`, `add_robot`, `add_camera`, `remove_camera`, `remove_robot`, `patch_scene_mjcf`, `replace_scene_mjcf`) invalidates it, and `load_state` then returns a structured error instead of writing a state vector whose indices now mean something else. Save a fresh checkpoint after mutating the scene |
| `set_joint_positions` | `positions` (dict or ordered list), `robot_name` (optional), `hold` (optional) - write `qpos` directly + run FK (teleport / set an initial pose, bypassing actuators). Kinematic only: a joint held by a position servo is pulled back toward the setpoint that servo already holds by the next `step`, and the success text names those joints. `hold=True` moves the matching position-servo setpoints with the pose so it survives stepping; a joint driven by a torque or velocity actuator is left alone, since its `ctrl` is not a pose, as is a joint a tendon couples to one `ctrl` (every stock gripper, and the `stretch3` telescoping arm), whose `ctrl` is in tendon units and drives several joints at once |
| `set_joint_velocities` | `velocities` (dict or ordered list), `robot_name` (optional) - write `qvel` directly (set an initial dynamic state) |
| `get_energy` | - |
| `get_sensor_data` | `sensor_name` (optional) |

The dict form of `set_joint_positions` / `set_joint_velocities` keys by joint
name; an unresolvable name refuses the whole write. `get_robot_state` reports
every joint of one robot by name (from a tool call); `robot_joint_names` is the
Python-only list.

## Actions

`send_action(action, robot_name=None, n_substeps=1)` writes actuator/joint targets and advances physics. `action` accepts either form:

| Form | Binding |
|------|---------|
| `{joint_or_actuator_name: value}` mapping | applied by name; unresolved keys are reported in an `unresolved_keys` JSON block so a caller can self-correct (no silent drop) |
| ordered numeric vector (`list` / `tuple` / 1-D `numpy` array) | bound positionally to `robot_action_keys(robot_name)` (the robot's actuator keys) in declaration order - the same convention `replay_episode` uses |

The vector form binds to `robot_action_keys` (the keys `send_action` resolves
and the order the `LeRobotDataset` recorder writes `action` in), not to
`robot_joint_names` - the two differ for robots with passive or mimic joints.
`n_substeps` is the number of physics steps the targets are held for: a
positive whole number (`np.int64(3)` and `3.0` are honored).

## Policy

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=50.0`, `action_horizon=8`, `n_steps=None`, `seed=None`, `async_rtc=None`, `rtc_inference_timeout_s=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""`) |
| `list_policies_running` | - |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (optional; auto-resolves the sole robot like `run_policy`), `n_episodes=1`, `max_steps=300`, `success_fn=None`, `async_rtc=False`, `rtc_inference_timeout_s=None`, `video=None` |
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

When a policy runs, the simulation sets its output keys to the robot's
*action keys* (`set_robot_state_keys(robot_action_keys(robot_name))`), which
are actuator names, not always joint names. Provider name + `policy_config`
(a dict) or a pre-built `policy_object=`: see
[Policy providers - in simulation](../policies/overview.md#in-simulation).

`stop_policy` is honored at any point after `start_policy` returns, including
before the first frame; `list_policies_running` names the robots in flight on
every backend, and a backend that cannot enumerate its rollouts is refused
rather than reported idle.

The horizon is `duration` (seconds) or `n_steps` (`n_steps` wins when both are
set; `max_steps` is an alias). `action_horizon` is how many actions are
consumed from each chunk before the policy is re-queried; `0`, negatives,
floats and `nan` are refused at every entry point rather than clamped.
`run_multi_policy` also accepts per-robot mappings (`instructions={robot: text}`,
`action_horizon={robot: horizon}`).

The four **posture** flags in the same signature are held to a domain of their own. `fast_mode`, `reset_between`, `wbc_install_torque_control` and `async_rtc` each select one of two branches rather than scale a quantity - pace the loop at `control_frequency` or run it unpaced, reset the scene between episodes or carry the end state over, install the WBC torque shim for the call or leave the actuators alone, overlap inference with actuation or drain each chunk first - so there is nothing to clamp and no partial effect, and a value that is not a boolean is refused rather than read by truthiness. Every non-empty string is truthy, so `"false"`, `"no"`, `"off"` and `"0"` would select the posture the word asks to skip, while `0`, `""` and `[]` would take the other branch without being a declared spelling of it; read that way, `run_policy(fast_mode="false")` ran unpaced, `run_policy(n_episodes=2, reset_between=0)` started episode two from wherever episode one left the arm, and `run_policy(async_rtc="false")` reported `rtc_async_enabled=True` beside the background inference thread the caller had declined - each with `status="success"`. The domain is the shared `boolean_flag_error` one that the recording postures and the mesh wire schema already use (the wire schema refuses this same `fast_mode` field unless it is a `bool`), bound to the tool-error envelope through `SimEngine._validate_posture_flags` and checked ahead of robot resolution, so a refused call builds no policy and touches no scene. `run_policy` checks all four; `async_rtc=None` is its documented "resolve from the policy" spelling and is checked only when a value is supplied, while `eval_policy` declares `async_rtc` as a plain `bool` and refuses `None` with everything else. MuJoCo's `start_policy` checks `fast_mode` before the submit, for the reason its numeric knobs already do: a refusal produced on the worker is discarded with the future and the caller reads "started". The `run_policy` agent tool checks its own `fast_mode` before it starts the recording it was asked to make, so the facade's refusal cannot arrive after the dataset at `dataset_root` has been replaced with an empty one. Unlike the numeric knobs above, the check sits at the facades only: `PolicyRunner.run` takes these flags as the facades hand them and does not repeat it.

**Episode outcome.** `eval_policy` calls `success_fn(observation)` and
`evaluate_benchmark` calls `is_success(sim)` / `is_failure(sim)` after every
applied action; a criterion that raises is fatal (naming criterion, episode and
step), as is `run_policy`'s `stop_when`, because a success rate over
undetermined episodes is not a measurement. Verdicts are read with `bool()`,
so a `numpy.bool_` is accepted. Predicate-DSL clauses (`stop_when`, a benchmark
spec's `success` / `failure` / `dense_reward`) refuse non-finite numeric
kwargs at compile time; `staged_reward` nests (a stage's `reward` may be
another `staged_reward`) and each machine resets its own sub-terms per episode.

Pass `seed=` to `run_policy` / `start_policy` for a reproducible rollout: it
reseeds Python / NumPy / torch / cuDNN and forwards `policy.reset(seed=...)`;
`eval_policy` seeds per episode.

### Async-RTC chunk pipeline (latency masking)

`async_rtc` overlaps policy inference with action execution; the flag's three
values, the auto-enable rule and the deterministic inference delay are
documented once, in
[LeRobot Local - synchronous vs async chunk execution](../policies/lerobot-local.md#synchronous-vs-async-chunk-execution-in-sim).
`eval_policy` defaults to `async_rtc=False` so success rates stay
bit-reproducible. `rtc_inference_timeout_s` bounds a stuck inference (a
positive finite number of seconds, or `None`); a policy that returns no
actions on its first query ends the rollout with `status="error"` after that
one query on both paths, and a *prefetched* chunk that arrives empty gets one
synchronous re-query before erroring. The spec path (`evaluate_benchmark` /
`evaluate(spec=...)`) stays synchronous and declares an observed delay of `0`
before every inference.


**Telemetry.** Every `run_policy` result `{"json": {...}}` block carries six RTC fields so latency masking is provable from the payload, not the logs:

| Field | Meaning |
|-------|---------|
| `rtc_async_enabled` | Whether the overlap pipeline ran (the resolved `async_rtc`) |
| `rtc_chunks_acquired` | Chunks the rollout acquired (cold start + swaps + re-queries), counted on the synchronous path too |
| `rtc_prefetch_hits` | Seams where the next chunk was already computed (stall hidden) |

A healthy masked rollout shows `rtc_prefetch_hits` near the chunk count and
`rtc_prefetch_blocks == 0`.

`run_policy` returns a `{"json": {...}}` block beside the human-readable
`text`: `robot_name`, `policy`, `instruction`, `n_steps`, `elapsed_s`,
`stopped_early`, `stopped_reason`, `action_errors`, `video_path` /
`video_frames` / `video_fps`, the policy-binding flags and the policy-load
telemetry (`policy_load_time_s`, `policy_load_cache_hit`). At `n_episodes > 1`
the same call returns an aggregate that adds `total_steps`, `stopped_reasons`,
`video_paths` and per-episode `episodes` records; per-step action health is
reported per episode, not averaged.

### Watching a rollout: the `observer` lane

`run_policy(observer=...)` takes a read-only callable that receives one
`RunPolicyStarted`, one `RunPolicyStep` per completed `send_action`, and one
`RunPolicyEnded` - a second lane beside the backend's `on_frame` hook.

```python
from strands_robots.simulation.observers import RunPolicyStep

def watch(event):
    if isinstance(event, RunPolicyStep) and event.action_resolution != "full":
        print(event.applied_action_index, event.unresolved_action_keys)

sim.run_policy(robot_name="alice", policy_provider="mock", observer=watch)
```

The events (observer schema version **2**) report four things `on_frame`'s
`(step, obs, action)` cannot carry:

| Field | Why it is not derivable from `on_frame` |
|-------|------------------------------------------|
| `action_resolution` | The backend's per-key `send_action` verdict, normalised to `full` / `partial` / `none` / `unknown`. `partial` and `none` require a valid, complete per-key breakdown; a coarse backend error is `unknown` with empty explicit key tuples, because input keys are not proof of what reached physical state. Coarse steps remain in `action_errors` and result text but are excluded from aggregate action-rate denominators rather than counted as physical misses. |
| `observation_is_chunk_reused` | Narrow chunk-position signal: `true` only for a later action using the same chunk-start snapshot. It is not authoritative freshness, because the first action after an async prefetch swap can already use an old snapshot. |
| `observation_age_steps` | Authoritative nonnegative age in control-step terms: completed rollout action attempts since the snapshot was sampled. Sync chunks report their chunk index. Async chunks carry the prefetch sample's remaining-old-chunk-attempt count across the swap and add the new chunk index. An active recording refreshes every step and reports `0`. On an `unknown` action resolution this field does not claim physical advancement. |
| `legacy_hook_outcome` | What the backend's hook did - `ok`, `cancelled`, `recording_error`, `error`, or `absent`. |


`event_seq` is dense and 0-based within one `run_id`; `monotonic_ns` orders the
stream and `utc_ns` derives from one rollout anchor. The lane is additive (it
adds only `observer_failures` to the result), contained (an observer's
`Exception` or `CooperativeStop` never changes the rollout outcome;
`KeyboardInterrupt`, `SystemExit`, `GeneratorExit` and `CancelledError`
propagate), borrowed (`observation` / `action` are the hook's own objects -
snapshot synchronously, do not retain) and not isolated (dispatch is
synchronous on the rollout thread, so a slow observer slows the robot). Scope:
`run_policy` and `PolicyRunner.run`; `eval_policy`, `evaluate_benchmark` and
`run_multi_policy` carry no observer yet.

`eval_policy` accepts the same `video={...}` config as `run_policy` but writes
one MP4 per episode (`eval.mp4` -> `eval_ep0.mp4`, ...).

**Benchmarks.** `evaluate_benchmark`, `list_benchmarks`,
`register_benchmark_from_file` and `register_builtin_benchmarks` score a policy
against a `success` / `failure` / `dense_reward` spec.
`register_builtin_benchmarks()` is opt-in and ships `go2_walk_forward`
(succeed past `x = 2 m`, fail on topple or height collapse, dense
velocity-tracking reward); `builtin_benchmark_specs()` returns the spec dicts
to fork.

## Recording

| Action | Notes |
|--------|-------|
| `start_recording(repo_id, task="", fps=30, ...)` | LeRobot v3 (parquet+MP4); requires `[lerobot]` extra |
| `save_episode()` | Flush the current rollout as one episode; call once per `run_policy` to record N episodes instead of one merged episode |
| `stop_recording(push_to_hub=False, bucket=None, run_id=None)` | Finalise dataset (flushes any trailing rollout) |
| `get_recording_status` | Episode, frame count, output dir |
| `start_cameras_recording(...)` | Plain MP4 via imageio-ffmpeg; `[sim-mujoco]` only, no lerobot |
| `stop_cameras_recording` / `get_cameras_recording_status` | - |

## Randomize

| Action | Key params |
|--------|-----------|
| `randomize` | `randomize_colors=True`, `randomize_lighting=True`, `randomize_physics=False`, `randomize_positions=False`, `position_noise=0.02`, `color_range=(0.1,1.0)`, `friction_range=(0.5,1.5)`, `mass_range=(0.5,2.0)`, `seed=None` |

Destructive - writes into model arrays. Recompile scene to undo.

## Registry

| Action | Notes |
|--------|-------|
| `list_urdfs` | Built-in robot table, plus a `Registered URDFs:` section naming every `register_urdf` asset and whether it resolves |
| `register_urdf(name, path)` | Register additional asset - it is named by `list_urdfs` from then on |
| `get_features(robot_name=None)` | Joint / actuator / camera / robot names of the scene (scoped to one robot with `robot_name`) - the source of truth for the action keys a policy must emit, and the feature schema used for recording |

When a policy's action keys resolve to no actuator, `run_policy` fails fast
naming `get_features(robot_name=...)` as the way to inspect the expected keys.

## See also

- [World building](world-building.md) - composing scenes.
- [Domain randomization](domain-randomization.md) - `randomize` distributions.
- [Policy providers](../policies/overview.md) - `policy_config` / `policy_object` forms.
- [LeRobot Local](../policies/lerobot-local.md) - `async_rtc`, RTC and the deterministic inference delay.
- [Architecture](../architecture.md)
