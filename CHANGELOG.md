# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased - PR #85 (MuJoCo backend remediation)

### Breaking

These changes tighten the MuJoCo AgentTool contract. Legacy callers that
silently worked by accident will now receive a clear error instead:

- **Router input validation**: The ``_dispatch_action`` router rejects any
  top-level parameter that isn't declared on the target method. Passing
  ``step(num_steps=5)`` (wrong name) or ``set_gravity(device="mps")``
  (stray kwarg) now errors with *"Unknown parameter X for action Y.
  Valid: [...]"* instead of silently dropping the value. Methods whose
  Python signature includes ``**kwargs`` (e.g. ``add_object``) keep their
  pass-through semantics.
- **Missing required args**: produce *"Action X requires parameter Y."*
  instead of a raw Python ``TypeError``.
- **Vector dimension validation**: ``position``, ``target``, ``origin``,
  ``force``, ``torque``, ``gravity``, ``direction``, ``point``, ``orientation``
  (quaternion), and ``color`` (rgba) all validated for length + numeric
  dtype before reaching numpy/MuJoCo.
- **Camera orientation**: ``add_camera(target=[x,y,z])`` is now honoured
  by baking ``xyaxes`` into the MJCF ``<camera>``. Previously the target
  was silently dropped and every custom camera rendered a default view.
  Degenerate case (``target == position``) errors.
- **Render camera validation**: ``render(camera_name="missing")`` errors
  with *"Camera 'missing' not found."* instead of silently falling back
  to the free camera while claiming to render from the named one.
- **Raycast zero-direction guard**: ``raycast(direction=[0,0,0])`` now
  errors with *"direction vector is zero-length"*. Previously MuJoCo's
  C-level ``mj_ray`` would abort the Python process.
- **apply_force requires a non-zero vector**: passing neither ``force``
  nor ``torque`` (or both zero) errors. Previously the call silently
  succeeded with no effect.
- **step(n_steps<0)** rejected (previously it corrupted ``step_count``).
- **Negative mass / timestep / size** rejected per shape; previously
  ``set_body_properties(mass=-1)`` and ``set_timestep(-0.01)`` silently
  succeeded.
- **Plane objects auto-static**: ``add_object(shape="plane")`` now forces
  ``is_static=True`` (planes are infinite in MuJoCo). Explicit
  ``is_static=False`` on a plane is a hard error.
- **Duplicate camera name** rejected. Previously a second ``add_camera``
  with an existing name silently overwrote the registry entry while
  leaving the old camera in the XML - ghost behaviour. Use
  ``remove_camera`` + ``add_camera`` to replace.
- **stop_policy(robot_name='')** errors with *"stop_policy requires
  'robot_name'."* instead of silently matching the first robot.
- **eval_policy** requires an explicit ``robot_name``. Default
  ``n_episodes`` lowered from 10 to 1.
- **register_urdf** validates the path: file must exist, be a file, and
  be readable. Previously bad paths were cached and blew up later.

### Recording backend split

- ``start_recording`` (LeRobotDataset: parquet + per-camera MP4) still
  requires the ``[lerobot]`` extra. Its error message when lerobot is
  missing now points callers at ``start_cameras_recording`` for plain
  MP4 (which runs under ``[sim-mujoco]`` alone via imageio-ffmpeg).
- No API change - the fix is informational.

### Resource hygiene

- ``destroy()`` and ``cleanup()`` now close renderers on the main thread
  and empty the TLS cache. Previously each ``create_world/destroy``
  cycle leaked one ``mujoco.Renderer`` + its GL context (~33 MB per
  cycle measured). Worker-thread renderers still release themselves on
  thread teardown (we avoid cross-thread ``close()`` to prevent
  ``cgl.free()`` SIGSEGVs on macOS).
- ``get_mass_matrix`` and ``get_contacts`` run ``mj_forward`` first so
  values are valid immediately after a ``reset`` or ``add_robot``
  (previously returned stale / uninitialised memory).

### Concurrency guards

Write-mutations are now refused while a policy is running on any robot
in the world. Previously these could race the policy worker thread and
produce undefined behaviour or SIGSEGV:

    reset, set_gravity, set_timestep, set_joint_positions,
    set_joint_velocities, apply_force, set_body_properties,
    set_geom_properties, load_state, randomize, move_object

The error now lists *which* robot(s) are active so the LLM can
``stop_policy`` on each without guessing: *"Cannot 'X' while a policy
is running on 'armA', 'armB'. Stop it first: action='stop_policy'."*

### Concurrent per-robot policies (GH #114)

Multiple ``start_policy`` calls on *different* robots now run
concurrently. MuJoCo physics is still serialized via ``self._lock``
(``mj_step`` and ``ctrl[]`` writes are not thread-safe for concurrent
mutation), but each policy owns a disjoint slice of ``data.ctrl[]`` so
two VLA arms can operate in the same scene without semantic conflict.

- ``start_policy("armA")`` + ``start_policy("armB")`` both succeed.
  Second call no longer hits a global "policy already running" gate.
- ``start_policy`` on the *same* robot while its policy is active
  still errors (unchanged).
- ``remove_robot("X")`` now gracefully stops X's own policy before
  removing, instead of requiring a prior ``stop_policy("X")``. Still
  errors if a *different* robot has an active policy (XML round-trip
  invalidates cached IDs everywhere).
- New action ``list_policies_running`` returns the names of robots
  with live policies. Prunes completed Futures as a side-effect.
- Completed policy Futures are no longer retained forever in
  ``_policy_threads`` (GH #120 companion fix).

### Error message consistency

- All "no world" paths return the same string:
  *"No world. Call create_world (or load_scene) first."*
- Unknown-name errors use a uniform ``<Kind> 'X' not found.`` shape
  (Robot / Object / Body / Geom / Joint / Sensor / Camera / Checkpoint).
- ``stop_recording``, ``stop_cameras_recording``, ``stop_policy``,
  ``close_viewer`` are now **idempotent**: calling them when nothing
  is running returns ``status="success"`` with a *"Was not ..."* message
  so callers can invoke them unconditionally.
- ``get_recording_status`` returns success in every lifecycle state
  (no world / not recording / recording).

### Deprecations

- **add_robot name-as-registry fallback**: passing ``name="my_bot"``
  without ``urdf_path`` or ``data_config`` used to resolve ``my_bot`` in
  the model registry. This now fires a ``DeprecationWarning``. Use
  ``add_robot(name="...", data_config="<registry_key>")`` instead. Will
  be removed next major release.

### New / extended actions

- ``forward_kinematics(body_name="X")`` filters to a single body.
- ``get_features(robot_name="X")`` filters to a single robot's joints
  and actuators.
- ``set_geom_properties(geom_name="X")`` accepts the bare object name
  as an alias for the injected ``"{name}_geom"``.
- ``render_all`` flags cameras whose frame has near-zero pixel variance
  (``"⚠️ camera 'X': image appears empty (variance < 1)"``).
- ``render_depth`` surfaces MuJoCo's one-time ``ARB_clip_control``
  warning in the response text on macOS, so the LLM knows when depth
  accuracy is reduced.
- ``render`` / ``render_depth``: width/height validated up front;
  oversized requests get a plain-English message naming the actual
  framebuffer cap (``<global offwidth=...>``) instead of MuJoCo's raw
  error.
- ``run_policy`` / ``start_policy``: accept optional ``n_steps``
  (primary) or legacy ``max_steps`` as an alternative to
  ``duration``+``control_frequency``. ``duration = n_steps /
  control_frequency`` when ``n_steps`` is set.
- **New ``list_policies_running``** action returns the names of robots
  with a live policy - pairs with the new concurrent-policy support
  (see *Concurrent per-robot policies* above).
- ``randomize(randomize_physics=True)`` now reports per-body mass scales
  and per-geom friction scales in the response (not just range
  endpoints).
- ``get_contacts`` resolves unnamed geoms to
  ``"<body_name>/geom_<id>"`` so contact pairs are always human-readable.
- ``get_sensor_data(sensor_name="X")`` on a model with no sensors now
  distinguishes *"Sensor 'X' not found. Model has no sensors."* from
  the generic "no sensors in model" success.

### Tests

- New: ``tests/simulation/mujoco/test_agenttool_contract.py`` - ~50
  tests that lock in router validation, tool_spec ↔ method parity,
  unified error messages, idempotent stop family, ``mj_forward`` before
  reads, render-dim validation, feature filters, camera duplicate
  policy, plane auto-static, policy horizon unification, and more.
- New: ``tests/simulation/mujoco/test_renderer_hygiene.py`` - 4 tests
  asserting TLS cache is emptied on ``destroy``, renderer reuse works
  for identical ``(w,h)``, and ``create_world`` after ``destroy``
  rebuilds cleanly.
- New: ``tests/simulation/mujoco/test_recording_backends.py`` - 2 tests
  (one skipped when ``lerobot`` IS installed) pinning the
  MP4-without-lerobot backend.
- New: ``tests/simulation/mujoco/test_input_validation.py`` - 11 tests
  for step/raycast/apply_force validation.
- New: ``tests_integ/test_resource_hygiene.py`` - 3 integration tests
  (require ``psutil``): 50 create/destroy cycles grow RSS < 50 MB; 500
  renders at fixed dims grow RSS < 100 MB; TLS cache cleared on destroy.

Test count: **256 → 362** (+106 new regression tests), zero
regressions. ``hatch run lint`` (ruff + mypy) clean across 102 source
files.
