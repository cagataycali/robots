# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased - mesh security hardening (Zenoh-native)

### Mesh wire authentication and authorisation now ride on Zenoh built-ins

Replaces the hand-rolled application-layer crypto (HMAC envelope,
nonce cache, per-peer identity, token bucket — ~1700 lines) with the
Zenoh primitives that ship out of the box: mTLS, per-key-expression
ACLs, multicast-off discovery, namespace routing isolation, and
per-key rate / size caps. Net effect: stronger guarantees, smaller
mesh tree, every claim verified against a live two-peer Zenoh
session in `tests/mesh/test_zenoh_transport_security.py`.

### Wire layer (Zenoh built-ins)

| Concern | Built-in | Configuration |
|---|---|---|
| Peer identity | `transport/link/tls` (mutual TLS) | `STRANDS_MESH_TLS_{CA,CERT,KEY}` |
| Authorisation | `access_control` keyed on cert CN | `STRANDS_MESH_ACL_FILE` (operator-supplied) or permissive built-in default |
| Replay protection | TLS per-link sequence numbers | n/a (transport-level) |
| Fleet isolation | `namespace` prefix at routing layer | `STRANDS_MESH_NAMESPACE` (default `strands`) |
| Discovery | gossip-only by default | `STRANDS_MESH_MULTICAST=true` to opt in |
| Per-key rate cap | `downsampling` block | `STRANDS_MESH_CMD_RATE_HZ` (default 20) |
| Per-message size cap | `low_pass_filter` block | `STRANDS_MESH_MAX_CMD_BYTES`, `STRANDS_MESH_MAX_CAMERA_BYTES` |
| Session-count DoS bound | `transport/unicast/max_sessions` | `STRANDS_MESH_MAX_SESSIONS` (default 256) |

### Payload-semantic layer (`mesh/security.py`)

What survives at the application layer (467 lines, was 1176):

* `validate_command()` — action allowlist + per-action bounds
  (instruction length, duration, step counts).
* `is_safe_policy_host`, `is_safe_model_path`, `is_safe_policy_type`,
  `is_safe_policy_provider`, `is_safe_server_address` — payload
  allowlists that ACL cannot see (it gates topics, not contents).
* `LockoutError` / `ValidationError` — typed exceptions for the
  command dispatcher.

Removed entirely:

* `mesh/identity.py` (505 lines) — replaced by mTLS cert chain.
* `sign_envelope` / `verify_envelope` / `_NONCE_CACHE` /
  `_record_nonce` / `clear_replay_cache` — replaced by TLS link auth.
* `TokenBucket` / `consume_peer_token` / `enforce_peer_rate_limit` /
  `_PEER_RATE_LIMITS` — replaced by `downsampling`.
* `pin_peer_identity` / `drop_peer_identity` /
  `known_peer_identities` — replaced by cert provisioning + ACL.
* `STRANDS_MESH_PSK`, `STRANDS_MESH_REQUIRE_AUTH`,
  `STRANDS_MESH_REPLAY_WINDOW`, `STRANDS_MESH_PEER_RATE`,
  `STRANDS_MESH_REQUIRE_PEER_IDENTITY`, `STRANDS_MESH_PEER_IDENTITY`,
  `STRANDS_MESH_PEER_KEY*` env vars.

### Zenoh 1.x quirks discovered and fixed

Empirical red-team testing against the live Zenoh 1.9 wheel
uncovered four bugs in our own first-pass config that this PR fixes:

1. **`low_pass_filter` `interfaces` semantics.** Earlier revisions
   of this PR enumerated every local NIC via psutil (with a hardcoded
   fallback list `["lo", "lo0", "eth0", "en0", "en1", "wlan0"]`).
   Cross-checking against `zenoh/src/net/routing/interceptor/low_pass.rs:84-91`
   showed that an absent `interfaces` field maps to
   `SubjectProperty::Wildcard` (matches every link), so the
   enumeration was both unnecessary AND a silent-bypass footgun on
   hosts with non-canonical NIC names (`enp0s3`, `wlp2s0`, `cni0`,
   `wg0`). The builder now omits `interfaces` by default so the cap
   applies fleet-wide; operators with a specific need to scope to a
   subset of NICs supply `STRANDS_MESH_FILTER_INTERFACES`. The
   transitive `psutil` import is gone.
2. **`<namespace>/*/cmd` patterns never matched.** Zenoh strips the
   namespace before checking key_exprs against the user-side key.
   All filter / ACL globs now use `**/cmd` style patterns.
3. **`access_control` requires `enabled: true`.** Without it the
   block parses but silently disables the gate. The ACL loader now
   hard-rejects any file that omits it.
4. **`cert_common_names` matches LITERAL CNs only** — globs do not
   work in Zenoh 1.x. The default ACL is therefore permissive (any
   CA-signed peer); operators wanting per-role enforcement supply
   `STRANDS_MESH_ACL_FILE` enumerating each peer's exact CN.
   `examples/mesh_acl_example.json5` is the canonical template.

### Surviving payload-semantic protections

* HITL interrupt for `emergency_stop` / `broadcast` in
  `tools/robot_mesh.py` — declined approvals do not consume a rate-
  limit slot.
* Per-action sliding-window rate limit on the LLM tool.
* Client-side `validate_command` on `Mesh.send` / `Mesh.broadcast`
  (so programmatic callers go through the same gate).
* `Mesh._on_response` `responder_id` binding for point-to-point RPC
  scope.
* `STRANDS_MESH_OVERRIDE_CODE` second factor on
  `_resume_lockout` (constant-time compare).

### Audit log (`mesh/audit.py`)

Independent of wire auth; kept as the forensic line of evidence:

* Per-record HMAC-SHA256 under `STRANDS_MESH_AUDIT_PSK`.
* Per-peer monotonic seq counters with sidecar persistence
  (`mesh_audit.seq.json`) and cross-process `fcntl.flock`.
* `O_NOFOLLOW` create + symlink reject on the audit file and seq
  sidecar.
* Bounded rotation (`STRANDS_MESH_AUDIT_MAX_BYTES`,
  `STRANDS_MESH_AUDIT_MAX_FILES`).
* `verify_audit_integrity` reads rotated logs in addition to the
  active one and walks every peer's seq cursor independently;
  records that fail signature verification do NOT advance the
  cursor (so a forged record cannot mask a real gap).
* `AuditPSKDegradedError` raised on EITHER PSK transition mid-run
  (signed -> unsigned OR unsigned -> signed). The unsigned->signed
  direction is refused symmetrically because a verifier cannot
  distinguish a legitimate PSK rollout mid-run from an attacker
  briefly clearing the PSK to forge unsigned records and then
  restoring it. Restart the process to transition deliberately.

### IoT path (`mesh/iot/provision.py`)

Hardened independently of the Zenoh refactor:

* AWS Root CA1 SHA-256 pin, accepting a tuple of pins so a future
  CA rotation can ship one PR ahead of the AWS rollout.
* `STRANDS_MESH_DISABLE_CA_PIN` break-glass NEVER applies to the
  on-disk re-use path — a rogue CA from a prior compromised run
  cannot be silently re-used.
* Per-receive timeout via custom `HTTPSHandler` (no global
  `socket.setdefaulttimeout` foot-gun).
* Thing-name regex tightened to `^[a-zA-Z0-9_-]{1,128}$` (alphanumerics, dash, underscore; length 1-128). Rejects path-traversal (`..`, `/`), NUL, whitespace, and non-ASCII.
* `iot/camera_offload.py` routes through public `Mesh.publish`.

### New env vars

| Var | Default | Purpose |
|---|---|---|
| `STRANDS_MESH_AUTH_MODE` | `mtls` | `mtls` (prod) or `none` (dev only). `none` ALSO requires `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` -- ERROR-level log at every session open. |
| `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE` | unset | Second-factor opt-in for `auth_mode=none`. Accepts `1`/`true`/`yes`. Refused otherwise -- prevents silent wire-auth disable from typos / forgotten env / leaked CI fixtures. |
| `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` | unset | Acknowledge the dev/lab posture of `mtls + permissive default ACL`. When unset, this combination logs at ERROR on every session open (F8-D). When set to `1`/`true`/`yes`, the log downgrades to INFO. Production deployments should ship a literal-CN `STRANDS_MESH_ACL_FILE` instead. |
| `STRANDS_MESH_TLS_CA` / `_CERT` / `_KEY` | unset | mTLS material; required when auth_mode=mtls |
| `STRANDS_MESH_ACL_FILE` | unset | Operator-supplied JSON5 ACL with literal-CN enumeration |
| `STRANDS_MESH_NAMESPACE` | `strands` | Fleet routing prefix |
| `STRANDS_MESH_MULTICAST` | `false` | Gossip-only by default |
| `STRANDS_MESH_MAX_SESSIONS` | `256` | Unicast session DoS bound |
| `STRANDS_MESH_CMD_RATE_HZ` | `20.0` | `downsampling` cap on `**/cmd` |
| `STRANDS_MESH_SAFETY_RATE_HZ` | `2.0` | `downsampling` cap on `**/safety/**` (R21 -- bounds novel-`t` flood vector) |
| `STRANDS_MESH_MAX_CMD_BYTES` | `16384` | `low_pass_filter` cap on `**/cmd` |
| `STRANDS_MESH_MAX_CAMERA_BYTES` | `1048576` | `low_pass_filter` cap on `**/camera/**` |
| `STRANDS_MESH_MAX_SAFETY_BYTES` | `4096` | `low_pass_filter` cap on `**/safety/**` (R21 -- jumbo-frame DoS bound) |
| `STRANDS_MESH_CAMERA_DISABLED` | `false` | Privacy kill switch -- when `true`, `Mesh._publish_cameras_once` short-circuits before any frame is built |
| `STRANDS_MESH_FILTER_INTERFACES` | unset (wildcard) | Optional comma-separated NIC allowlist for the `low_pass_filter` rules. Unset means "every link" (Zenoh's `SubjectProperty::Wildcard`). |
| `STRANDS_MESH_RESUME_FRESHNESS_S` | `60` | Maximum age (seconds) of a resume envelope before rejection as stale |
| `STRANDS_MESH_RESUME_FORWARD_SKEW_S` | `5` | Maximum forward clock skew (seconds) tolerated in resume envelope timestamps |
| `STRANDS_MESH_RESUME_REPLAY_CACHE_MAX` | `4096` | Maximum entries in the per-receiver resume proof_nonce replay cache (LRU eviction) |

All three resume-hardening env vars are parsed via
``_parse_positive_float_env`` / ``_parse_positive_int_env`` helpers in
``strands_robots.mesh.core``. Malformed values (non-numeric or
negative) log a WARNING and fall back to the documented default
instead of failing module import or silently disabling the replay
cache. Pinned by ``tests/mesh/test_resume_env_validation.py``.

### Tests

* **839 mesh tests pass** (4 skipped with documented reasons -- 2 ACL fanout cases tracked in #200, plus 2 mTLS-handshake tests gated on the cryptography PyPI dep).
* New: `tests/mesh/_pki.py` — ephemeral CA + leaf cert helper.
* New: `tests/mesh/test_zenoh_transport_security.py` — 11 tests, 9 active, run
  against real two-peer `zenoh.open()` sessions covering namespace
  isolation, downsampling / low_pass_filter enforcement, mTLS
  handshake rejection of rogue CA, ACL drop of unknown CN.
* New: `tests/mesh/test_validate_command.py`, `test_zenoh_config.py`,
  `test_acl_config.py`, `test_session_config.py`,
  `test_audit_integrity.py`, `test_iot_ca_pin.py`,
  `test_iot_policy_scope.py`, `test_camera_acl.py`,
  `test_robot_mesh_security.py`, `test_application_security.py`,
  `test_bridge_dedup.py`.

### Migration

Existing deployments under `STRANDS_MESH_PSK`:

1. Provision a CA + per-peer leaf certs (or reuse the AWS IoT certs
   already in `~/.strands_robots/iot/`).
2. Set `STRANDS_MESH_TLS_{CA,CERT,KEY}` on every peer.
3. (Optional) Drop `examples/mesh_acl_example.json5` at
   `STRANDS_MESH_ACL_FILE` and edit the `cert_common_names` lists
   with your fleet's exact CNs.
4. Drop all `STRANDS_MESH_PSK` / `STRANDS_MESH_PEER_KEY*` /
   `STRANDS_MESH_REPLAY_WINDOW` / `STRANDS_MESH_PEER_RATE` env vars.

### Breaking: `policy_provider` is required on `execute` / `start`

Pre-PR, `policy_provider` defaulted to `"mock"` inside the dispatcher.
`validate_command` (`mesh/security.py`) now REJECTS any `execute` /
`start` command without an explicit `policy_provider`. This is a
defence-in-depth gate: silent default-to-mock would let an
authenticated peer steer the robot at the no-op policy and get arbitrary
text echoed back without the operator noticing.

Migration for programmatic callers using `Mesh.send` / `Mesh.broadcast`:

```python
# Pre-PR (silently defaulted to mock):
mesh.send(target, {"action": "execute", "instruction": "do thing"})

# Post-PR (must specify):
mesh.send(target, {
    "action": "execute",
    "instruction": "do thing",
    "policy_provider": "mock",     # or "lerobot_local", "groot", etc.
})
```

Operators using the LLM tool surface (`tools/robot_mesh.py`) are unaffected --
the tool already supplies `policy_provider` from its own arg.

---

Dev / lab environments without PKI run `STRANDS_MESH_AUTH_MODE=none` AND `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1`
to keep plain-TCP behaviour; the mesh logs a WARNING at session
open.

## Unreleased - #178 (LiberoOffScreenRenderEngine retired)

### Removed: ``LiberoOffScreenRenderEngine`` simulation backend (BREAKING)

After PR #184 made ``MuJoCoSimEngine`` byte-equivalent to upstream LIBERO
(model-level inertias, ``mj_step`` divergence 0 over 200+ substeps, mean
``success_rate=0.92`` vs offscreen ``0.72`` on libero-10/SCENE5),
``LiberoOffScreenRenderEngine`` has no functional reason to exist. It is
deleted entirely.

What is gone:
- **Deleted**: ``strands_robots/simulation/libero_offscreen_render/``
  (entire package, ~700 LoC).
- **Deleted**: ``"libero_offscreen_render"`` registry entry in
  ``strands_robots.simulation.factory`` and its aliases
  ``"libero_offscreen"`` and ``"libero_osr"``.
- **Deleted**: ``LiberoAdapter._on_episode_start_offscreen`` and the
  ``hasattr(sim, "setup_libero_task")`` dispatch branch in
  ``LiberoAdapter.on_episode_start``. The unified ``MuJoCoSimEngine``
  path is the only path now.
- **Deleted**: ``LiberoAdapter.is_success`` no longer delegates to
  ``env.check_success`` on ``OffScreenRenderEnv``-backed engines (no
  such engines exist anymore). It now always evaluates the BDDL
  predicate tree, hardened in #170 / #173 / #175 to match upstream's
  ``check_ontop`` / ``check_contact`` semantics.
- **Deleted**: ``STRANDS_LIBERO_PREDICATE_LOG`` and
  ``STRANDS_LIBERO_PREDICATE_LOG_MAX`` env vars (the BDDL ↔
  ``env.check_success`` disagreement diagnostic; no offscreen env
  to compare against). The ``_walk_predicate_tree`` helper is kept
  for any future BDDL-evaluator debugging.
- **Deleted**: ``tests/simulation/libero_offscreen_render/`` (3 unit
  test files).
- **Rewrote**: ``tests_integ/benchmarks/libero/test_upstream_state_parity.py``'s
  ``test_state_observation_byte_equivalent_at_canonical_init`` to
  compare ``MuJoCoSimEngine`` directly against upstream's raw
  ``OffScreenRenderEnv`` (skipping the intermediate engine wrapper).
  Same coverage, less indirection.

Migration: rename the backend in any ``create_simulation()`` call.

```python
# Before
sim = create_simulation("libero_offscreen_render", ...)
# (also "libero_offscreen", "libero_osr")

# After
sim = create_simulation("mujoco", ...)
```

The ``mujoco`` backend now reaches ``success_rate >= 0.92`` on
libero-10/SCENE5 (vs ``0.72`` for the offscreen engine), so this is
strictly an upgrade for benchmark eval consumers.

Out of scope: ``examples/libero_mujoco.py`` in
``strands-labs/robots-sim`` still has an ``--engine={mujoco,libero_offscreen_render}``
switch. A follow-up issue tracks updating it once this PR lands.

## Unreleased - PR #85 (MuJoCo backend remediation)

### MJCF builder refactor: string-concat -> MjSpec AST (closes #121, #122-#126)

The ``MJCFBuilder`` string-concat path and the ``scene_ops`` XML-round-trip
machinery (~700 lines total) are replaced by direct manipulation of
``mujoco.MjSpec`` - the editable MJCF AST shipped with MuJoCo 3.2+.

What changed under the hood:
- **New module** ``strands_robots/simulation/mujoco/spec_builder.py``. The
  ``SpecBuilder`` class owns scene construction + mutation (``build``,
  ``add_object``, ``remove_body``, ``add_camera``, ``remove_camera``,
  ``attach_robot``, ``from_mjcf_string``, ``from_file``).
- **Deleted**: ``strands_robots/simulation/mujoco/mjcf_builder.py`` (273
  lines of f-string MJCF and the ``_camera_xyaxes_from_target`` helper).
- **Rewrote**: ``strands_robots/simulation/mujoco/scene_ops.py`` from
  ~980 lines of tmpdir + ``mj_saveLastXML`` + ``ElementTree`` round-trips
  down to ~295 lines that go through ``spec.recompile(model, data)``.
- **Bumped**: ``mujoco>=3.0.0`` -> ``>=3.2.0`` in ``pyproject.toml`` so
  ``MjSpec`` is always available. Current hatch env runs 3.8.0.

Agent-visible wins:
- **New action** ``patch_scene_mjcf(ops=[...])`` - apply a list of
  structured ops (add_body, add_geom, add_site, set_body_pos,
  set_body_quat, delete_body) to the live spec atomically. Whole batch
  is rolled back from an XML snapshot if any op fails; one
  ``spec.recompile()`` for the whole batch, so qpos/qvel for unchanged
  joints are preserved. Narrower surface than ``replace_scene_mjcf``
  but much cheaper for surgical edits (no full-scene XML round-trip).
- **New action** ``replace_scene_mjcf(xml=...)`` - atomically replace the
  whole scene with agent-authored MJCF. Validated by actually compiling
  it, so ``<tendon>``, ``<equality>``, ``<pair>``, custom solref/solimp,
  sites, hfield, etc. all work without needing new ``SimObject`` shape
  vocabulary. On malformed XML returns a clean error dict (no process
  abort).
- **``ellipsoid`` shape** now works in ``add_object`` - it's a free
  bonus MuJoCo geom type the string-concat builder rejected.
- **Camera orientation** uses ``quat`` (computed via
  ``mujoco.mju_mat2Quat``) instead of a hand-rolled ``xyaxes`` string.
  Compiled ``cam_mat0`` is numerically identical within ~4e-7.
- **``spec.recompile(model, data)``** preserves existing joint qpos/qvel
  for unchanged joints automatically - no manual "copy state by name"
  loop. Object freejoints added post-compile get initialised to the
  body's ``pos``/``quat``.
- **No more XML injection surface**: names go straight into MjSpec which
  validates them itself, so the old ``_sanitize_name`` regex gate +
  fuzz test are no longer needed.

Downstream API is unchanged: ``add_object``, ``add_robot``, ``remove_object``,
``remove_robot``, ``add_camera``, ``remove_camera``, ``load_scene`` all keep
their tool-action signatures. Tests that asserted on exact XML strings
were rewritten to assert on compiled ``MjModel`` properties (``cam_mat0``,
``mj_name2id``) so they are representation-agnostic.

Known constraint: ``remove_robot`` now rebuilds the scene from scratch
(drops joint qpos state) rather than going through ``spec.delete()`` on
attached bodies. This sidesteps a MuJoCo 3.8 double-free bug where
``spec.delete(attached_body)`` + interpreter shutdown crashes. Trade-off
is documented in ``scene_ops.eject_robot_from_scene``.

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

### Policy-hook robustness (GH #117)

``PolicyRunner.run`` previously caught *all* ``on_frame`` exceptions at
WARN level and kept iterating. A recording hook with a typo'd observation
key would log 500 lines and produce an empty dataset. Now we count
*consecutive* failures and abort the episode after a threshold (default
5, tunable via new ``max_onframe_failures`` kwarg).

- A single transient failure still logs + continues; counter resets on
  the next successful call.
- ``N`` consecutive failures raise ``RuntimeError`` so ``run()`` returns
  ``status='error'`` with a clear message, preventing silent dataset
  corruption.

### Cleanup graceful shutdown (GH #116)

``Simulation.cleanup()`` no longer races the policy worker. Previously
cleanup set ``self._world = None`` and called ``executor.shutdown(wait=False)``
nearly simultaneously - a policy still inside ``mj_step`` segfaulted on
freed arrays. Now cleanup:

1. Signals every live policy to stop (``policy_running = False``).
2. Awaits each outstanding Future with a bounded timeout (default 5s,
   overridable via new ``cleanup(policy_stop_timeout=...)`` kwarg).
3. Only AFTER workers unwind do we null ``self._world`` and tear down
   renderers / viewer / executor.

Wedged workers that don't stop in time get logged as a warning - cleanup
proceeds rather than hanging the host process on exit.

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
