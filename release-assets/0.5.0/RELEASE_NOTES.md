# strands-robots 0.5.0

**Three simulation backends behind one interface, locomotion as a first-class
capability, an agent that can point at a pixel - and a 104-PR campaign that went
through every public parameter asking one question: *if the library accepts this,
can it actually honor it?***

*805 pull requests / 806 commits since [v0.4.1](https://github.com/strands-labs/robots/releases/tag/v0.4.1),
measured at `ab141191` (2026-07-01 -> 2026-08-03). 70 features, 294 fixes, 327 test PRs, 90 docs.
`strands_robots` +40,475 lines across 160 files; the test suite went from 7,210 to
12,139 test functions (+68%), expanding to **18,536 passing tests**. Every visual
below was rendered headless in MuJoCo (`MUJOCO_GL=egl`) on an NVIDIA Jetson AGX Thor
against the exact source being tagged.*

---

## Highlights

| | What changed | Evidence |
|---|---|---|
| **Isaac Sim is in-tree** | A third backend joins MuJoCo and Newton behind the same `SimEngine` ABC. `pip install "strands-robots[isaac]"`, `Robot(..., backend="isaac")`. | 6 PRs, recording + EEF + GR00T actuation parity |
| **Locomotion, first class** | 4 terrain heightfields with a difficulty curriculum, 11 locomotion predicate/reward terms, 5 shipped benchmarks (go2 x3, g1, t1). | ![terrain](https://raw.githubusercontent.com/cagataycali/robots/artifacts/release-notes-0.5.0/release-assets/0.5.0/feat_terrain.png) |
| **Point at a pixel** | `get_world_point(camera, pixels)` grounds a pixel to a world coordinate; `move_to` / `set_gripper` / `rotate_wrist` are analytic primitives on shared mink IK. | ![grounding](https://raw.githubusercontent.com/cagataycali/robots/artifacts/release-notes-0.5.0/release-assets/0.5.0/feat_grounding.png) |
| **No policy could open a gripper** | On a tendon-driven gripper the actuator spelling - the one `robot_action_keys()` returns - wrote a raw `1.0` into a `[0, 255]` range. | ![gripper](https://raw.githubusercontent.com/cagataycali/robots/artifacts/release-notes-0.5.0/release-assets/0.5.0/fix_gripper.png) |
| **Robots were simulated wrong** | `<option>` is model-global and does not survive `spec.attach()`, so every solver setting a robot MJCF declared for itself was discarded. | ![options](https://raw.githubusercontent.com/cagataycali/robots/artifacts/release-notes-0.5.0/release-assets/0.5.0/fix_options.png) |
| **Adding a robot rewound the world** | `add_robot` reset the whole scene to give the new robot a clean start. | ![rewind](https://raw.githubusercontent.com/cagataycali/robots/artifacts/release-notes-0.5.0/release-assets/0.5.0/fix_rewind.png) |

---

## 1. Isaac Sim: the third backend

The Isaac Sim backend moved into this repository (#1156) behind the `[isaac]`
extra (#1155), so all three engines - MuJoCo, Newton, Isaac - now sit behind one
`SimEngine` ABC and one agent-tool surface.

```python
from strands_robots import Robot

r = Robot("panda", mode="sim", backend="isaac")   # same call as backend="mujoco"
```

Parity landed feature by feature rather than as a claim: LeRobotDataset recording
(#1552), an EEF state source for `LiberoAdapter` and `get_body_state` (#1811),
GR00T action actuation via a delta-EEF differential-IK controller (#1819), and the
LIBERO init-state arm qpos applied by mapping robosuite joint names onto the USD
articulation (#1832). It survives pip Isaac Sim 6.0.x (#1798), reports a URDF's own
joint names rather than USD's mangled forms (#1902), and refuses an entity name the
backend cannot address (#1845, #1847) - Isaac builds `prim_path` by interpolation, so
an empty name resolved to the container holding *every* robot.

**Backends are no longer allowed to disagree.** A large share of this cycle's fixes
are cross-backend parity: one `add_object` size / mass / colour domain on every
backend (#1861, #1859, #1856), one `create_world(difficulty=)` domain (#1857), one
scene-placement pose rule (#1853), one step count (#1869) and substep count (#1880),
one camera pose/fov domain (#1760, #1764). Several are pinned by AST parity tests, so
a fourth backend cannot ship without them.

## 2. Locomotion became a first-class capability

**Ground.** `create_world(terrain=...)` generates a heightfield - `"rough"`,
`"stairs"`, `"pyramid"`, `"slope"` (#1336, #1338, #1339, #1340) - with
`difficulty=` scaling peak elevation as a curriculum knob (#1344, advertised in the
tool spec by #1350). `get_ground_height(x, y)` queries the local surface (#1388).

A floating base is **seated on the local terrain surface**, not at an absolute z
(#1386). Measured on the panel above, standing clearance holds to four decimal
places while the ground beneath quadruples:

| `difficulty` | ground at origin | go2 base z | clearance |
|---|---|---|---|
| 1.0 | 0.04 m | 0.3871 m | **0.3471 m** |
| 2.5 | 0.10 m | 0.4471 m | **0.3471 m** |
| 4.0 | 0.16 m | 0.5071 m | **0.3471 m** |

**Reward/predicate DSL.** Eleven locomotion terms: `base_beyond_x` / `base_beyond_y`
/ `base_yaw_beyond` progress predicates, `base_tipped` / `base_below_z` failure
predicates, and `base_height` / `base_orientation` / `base_velocity` /
`base_velocity_tracking` / `base_lin_vel_z` / `base_ang_vel_xy` reward terms
(#1198-#1324). `base_height` and `base_below_z` measure clearance above *local*
terrain rather than absolute z (#1368, #1364) - on a heightfield those are different
questions.

**Benchmarks.** `register_builtin_benchmarks()` ships five: `go2_walk_forward`,
`go2_strafe_left`, `go2_turn_left`, `g1_walk_forward`, `t1_walk_forward`
(#1259, #1287, #1288, #1321, #1324).

**Floating-base state correctness** took ~15 fixes: a free joint is surfaced as a
structured `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel` rather than a
degenerate scalar (#1164, #1176, #1318), preserved in recorded datasets (#1172,
#1183), reported in the body frame (#1187), and joint index maps no longer shift
every joint after a floating base (#1139). Most recently, a Newton floating base is
no longer advertised as an actuator (#1913) - it appeared in `robot_action_keys`, so
a floating-base recording could be written and never replayed.

## 3. An agent can point at a pixel

Two features that change what an LLM can do without writing kinematics.

```python
frame = sim.render(camera_name="eye")            # the agent looks
pt = sim.get_world_point("eye", [[374, 257]])    # and names a pixel
sim.move_to(position=pt, orientation=[0, 1, 0, 0])   # the arm goes there
sim.set_gripper(state="close")
```

`get_world_point(camera, pixels)` (#1649) unprojects through the camera's own
intrinsics and depth. Measured on the panel above, against `get_body_state` ground
truth: **5.08 mm** in xy, and **0.4 mm** in z against the cube's visible top face -
a pixel resolves to the point on the surface the camera can see, which is what an
agent looking at a frame actually means.

`move_to` / `set_gripper` / `rotate_wrist` (#1654) are analytic primitives backed by
shared mink IK, with gripper classification resolved registry-first from per-robot
metadata rather than a name heuristic (#1660, #1662). The reach above converged with
a **0.56 mm** IK residual. `[sim-mujoco]` now declares the QP backend that solve
needs (#1683, #1788) - it was reaching a third-party transitive dependency.

Also: `run_policy(stop_when=...)` takes a semantic early-return predicate and
reports `stopped_reason` (#1656), and `harness_memory` records task solution traces
with global success rules and failure models (#1651).

## 4. The campaign: 104 PRs asking one question

The defining character of this release. **104 of the 294 fixes** refuse, honor or
validate a parameter that was previously accepted and then silently not applied.
They are not scattered paper cuts - they are one defect family, found by
systematically walking every public surface and asking whether a value the API
accepts can actually be carried out.

The family has a signature. A caller passes something the code cannot honor; nothing
raises; `status` reports `"success"`; and the robot does something else. The three
visuals above are the sharpest instances:

**A gripper no policy could open** (#1652). `send_action` resolved an action key by
actuator name *or* by a joint that actuator drives, and only the joint-name branch
applied the unit mapping. On the Panda's tendon gripper the same fully-open command
landed 255x apart:

| action key | pre-fix finger gap | post-fix |
|---|---|---|
| `{"actuator8": 1.0}` | **0.31 mm** - shut | 80.0 mm |
| `{"finger_joint1": 1.0}` | 80.0 mm | 80.0 mm |

Both returned `success`. The severity is which spelling matters:
`robot_action_keys("arm")` returns `['actuator1' ... 'actuator8']` - **only** the
actuator form. That is what a policy is fed, what a positional vector binds to, and
what names a dataset's action columns. No policy could open that gripper.

**Robots simulated under settings their own model rejects** (#1687). MuJoCo
`<option>` is model-global and is not carried across `spec.attach()`, so every
solver setting a robot MJCF declared for itself was discarded when `add_robot`
merged it into a scene. `panda.xml` declares `<option integrator="implicitfast"/>`;
the scene kept Euler. Under a held set-point with stiff position servos - the exact
case the declaration exists for - the arm **never came to rest**:

| | compiled integrator | residual jitter | tracking error |
|---|---|---|---|
| before | `mjINT_EULER` | **0.4655 rad/s** (never settles) | 0.024209 rad |
| after | `mjINT_IMPLICITFAST` | **0.0 rad/s** | 0.009418 rad |

This is not one asset's quirk. Of the 53 registry robots whose asset resolves
locally here, **42 declare an `<option>`**: `so100`, `aloha`, `shadow_hand` and
`robotiq_2f85` all declare `cone="elliptic" impratio="10"` - the standard recipe for
a gripper that must hold load - `unitree_go2` declares `impratio="100"`, and
`unitree_g1` and every Franka/UR/xArm declare `integrator="implicitfast"`. The scene now adopts a declared field
when the world has not set it, and leaves the caller's own `create_world` knobs
alone.

**Adding a robot rewound the world** (#1763). `add_robot` called `mj_resetData` on
the whole world to give the newly-merged robot a clean start. A parked arm and a
settled crate both went back to spawn - the crate to mid-air - under
`status="success"`, with the clock at zero. Its own docstring promised the opposite
("this preserves previously-created world state").

Fifty more in the same family, each with a measured consequence: `duration`,
`control_substeps`, `action_horizon` and `control_frequency` honored or refused
rather than clamped (#1638, #1639, #1718, #1700); every dataset recording rate that
would mislabel a capture refused across all four MP4/dataset entry points (#1664,
#1666, #1668, #1746, #1749, #1751, #1669); `randomize` / `set_obs_noise` ranges
(#1670); `patch_scene_mjcf` op keys and numeric fields (#1681, #1757, #1860); a
runtime setter's value surviving the next recompile (#1703); a geom resize
re-deriving the owning body's inertia (#1753); an object's inertia integrated from
its shape instead of a hard-coded constant (#1694); a latched wrench belonging to
the body it was applied to (#1698); a physics checkpoint refused once its model is
swapped (#1756); a non-finite action value refused before it poisons the shared
state vector (#1755); a non-string entity name reported instead of segfaulting the
process (#1773).

The same sweep ran outside `simulation/`: the hardware control loop
(#1700, #1710, #1718, #1724, #1725, #1727, #1728, #1730, #1734, #1737), teleop and
mesh (#1676, #1684, #1686, #1693, #1719, #1721, #1817), the agent tools
(#1706, #1713, #1720, #1782, #1813, #1816, #1821, #1834, #1836), the policy
providers (#1704, #1739, #1743, #1744, #1750, #1752, #1809, #1818, #1839, #1850),
and the shared guards themselves - a refusal must be able to quote what it refuses
without raising (#1876, #1879, #1881, #1890, #1898, #1904, #1907, #1908, #1910).

## 5. Remote inference and transports

- **`inference` client/server split** over WS-JSON (#1142): run a heavy policy on a
  GPU box and drive an arm from a laptop. Observations decode into writable arrays
  on both paths (#1152), the frame limit lifts for image observations (#1158), and
  the server publishes its port before reporting bound (#1250).
- **`lerobot_async` provider** (#1212) over lerobot's gRPC async transport, with
  `rename_map` forwarded to the remote `PolicyServer` (#1398) and the supported
  policy set sourced live from lerobot rather than a stale copy (#1458).
- **rosbridge transport** (#1110): `use_rosbridge` + `RosbridgeRobot`, verified live
  against NASA's Curiosity Gazebo model. ROS 2 actions and goal-level `navigate_to`
  landed in `use_ros` (#995).
- **Streaming datasets** are exported module-level (`stream_dataset`,
  `StreamingDatasetReader`, #1513) with `repo_type` forwarded (#1447), and
  `sync_dataset_to_bucket(root, bucket)` is lifecycle-independent (#1515).

12 policy providers now register: `mock`, `groot`, `lerobot_local`,
`lerobot_async`, `cosmos3`, `vera`, `wbc`, `wbc_gait`, `moveit2`, `curobo`,
`motionbricks`, `remote`.

## 6. Hardware path

Previously the least-exercised surface, and it showed. `cleanup()` closes the motors
bus and every camera (#1734); a failed `connect` closes every device it left
half-open (#1728); `stop()` is terminal for a robot that never fully connected
(#1737); a rollout is refused on a robot that has been shut down (#1730); a second
rollout cannot take a bus one already owns (#1725); a stop pressed during multi-second
bring-up actually stops the task (#1724); a policy's per-episode state resets at the
start of each task (#1712); and `pose_tool emergency_stop` **de-energizes the arm**
instead of reporting that it did (#1706) - it was a bare success return with a
"would require torque disable" comment.

SO-101 real-hardware support covers camera `fourcc` and sim-embodiment policies
(#1657), a leader arm is registered as a teleoperator rather than a follower robot
(#1674), and per-camera config keys are derived from lerobot's own dataclass fields
so a typo is refused and no field is unreachable (#1705).

## 7. Discovery, so an agent can find the surface

`describe()` now advertises every family - 17 PRs, covering world lifecycle and MJCF
editing, the robot registry, physics tuning and domain perturbation, benchmark
scoring, checkpoints and pose setting, background-policy lifecycle, plain-MP4
recording, teleoperation, the interactive viewer, multi-robot rollout, and physics
introspection (#1249-#1348). The MuJoCo backend exposes **76 agent-tool actions**.

Errors became navigable: an unknown robot, body, site, geom, sensor or camera now
returns a close match plus the canonical listing action (#1299, #1303, #1306, #1308,
#1352, #1436), and the registry ships **72 robots** (23 arms, 18 humanoids, 10
mobile, 9 hands, 5 mobile-manipulators, 4 bimanual, 2 aerial, 1 expressive) with
every alias documented (#1713, #1716).

## 8. Docs, examples, CI

45 documentation files and 76 example files changed. The `robots-sim` examples were
absorbed (#1282, #1280, #1278, #1545), a streaming data-loop notebook added (#1331),
and a MolmoAct2 collect-train-run example landed (#586).

CI gained four guards, each from a defect it would have caught: a changelog entry
written outside a fragment is refused (#1785, with fragments replacing log appends in
#1692), a PR whose base moved under the files it edits is refused (#1771), a gate
reads its own script from the base branch rather than the tree under review (#1793),
and a PR whose only approval came from its own pusher is named (#1921). Two security
gates route dangerous `lerobot_train` passthroughs through human-in-the-loop approval
(#1085, #1697), and `safe_join` rejects symlink traversal for untrusted clones (#1627).

---

## Upgrade notes

- **`add_robot` no longer resets the scene.** If you relied on it to zero a world,
  call `reset()` explicitly.
- **A robot's declared `<option>` is now adopted** when the world has not set that
  field. Rollouts against models declaring `implicitfast` / `elliptic` cones will
  differ from 0.4.1 - they are now integrated as the model asks. `create_world`'s
  own `timestep` / `gravity` always win.
- **Values previously accepted and ignored are now refused** with a structured
  error. If a call newly errors, it was not being honored before: the message names
  the parameter and the accepted domain.
- **`[isaac]` is a new extra**; `[sim-mujoco]` now declares its QP backend, so
  `move_to` works from a clean install.
- Python >= 3.12, `lerobot >= 0.6.0`.

## Verification

Rendered on an NVIDIA Jetson AGX Thor (sm_110, CUDA 13.0), torch 2.11.0+cu130,
mujoco 3.11.0, lerobot 0.6.2, `MUJOCO_GL=egl`.

Full gate at `ab141191`:

```
MUJOCO_GL=egl pytest tests    ->  18536 passed, 257 skipped in 508.32s
ruff check                    ->  All checks passed!
ruff format --check           ->  1048 files already formatted
mypy strands_robots tests ..  ->  Success: no issues found in 1045 source files
```

Every before/after panel runs **one script against two git worktrees** - the merge
parent of the fixing PR, and `main` - so the fix is the only variable. Each figure
self-audits before it is written: measured facts are re-derived from the run's JSON,
panels that must differ are checked for a differing-pixel floor, and panels that must
match are checked against renderer noise. The #1652 script asserts both directions -
that the two spellings' renders *differ* pre-fix and *agree* post-fix - which is how
a mis-aimed camera was caught rather than shipped.

Generating scripts and full-resolution PNGs:
[`release-assets/0.5.0/`](https://github.com/cagataycali/robots/tree/artifacts/release-notes-0.5.0/release-assets/0.5.0)

Full changelog: `git log v0.4.1..v0.5.0`, or
[`compare/v0.4.1...main`](https://github.com/strands-labs/robots/compare/v0.4.1...main).
