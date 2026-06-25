# VERA policy × strands-robots embodiments — in-depth coverage analysis

**Question:** does the `strands-robots` VERA provider work with all our robot
embodiments (68 robots, 8 categories)?

**Short answer:** **No — and it cannot, by design.** VERA ships **4 trained
embodiments** (pusht, mimicgen, allegro, droid). A VERA policy is a *trained
model pair* (video planner + Jacobian IDM) for a *specific* embodiment — you
can't point it at an arbitrary robot any more than you can run a GR00T checkpoint
trained for SO-100 on a Unitree G1. BUT there is also a **real wiring bug** that
stops even the 4 supported embodiments from driving our robots correctly. That
bug is fixable; the "all 68 robots" expectation is not (it's a model-availability
limit, not a code limit).

---

## 1. What VERA actually provides (the 4 trained embodiments)

From `vera/server/protocol/adapter_factory.py` (`_EMBODIMENTS`):

| VERA embodiment | action_space | action dims | views (cameras) | proprio | control_dt | ckpts |
|---|---|---|---|---|---|---|
| **pusht**   | `pos`             | 2 (planar x,y), no gripper | `image` (1) | — | 1/10 s | local (Wave-1) |
| **mimicgen**| `eef_delta`       | ~7 (6-DoF eef delta + gripper) | `agentview_image`, `robot0_eye_in_hand_image` (2) | eef_pos/quat/gripper | 1/20 s | Wave-1 (+WAN base) |
| **allegro** | `joint_position`  | 16 (hand joints) | `camera_0..11` (12) | q_robot | 1/15 s | Wave-2 (no ckpt yet) |
| **droid**   | `cartesian_delta` | 7 (6-DoF + gripper) | `varied_1`,`varied_2`,`hand` (3) | q_robot/eef_pos/quat/gripper | 1/15 s | Wave-2 (no ckpt yet) |

**Only pusht + mimicgen have released checkpoints (Wave 1).** allegro + droid are
code-present, checkpoint-absent (Wave 2). So *today* the provider can serve **2**
embodiments end-to-end.

---

## 2. Our 68 robots (registry/robots.json)

    arm: 22   humanoid: 18   mobile: 10   hand: 8   mobile_manip: 4
    bimanual: 3   aerial: 2   expressive: 1                         = 68

### Which of our robots could a VERA embodiment plausibly drive?

VERA is a **manipulation** policy (eef-delta / joint-position / planar push). It
has **no** locomotion / flight / whole-body planner. So:

| VERA embodiment | Natural strands-robots targets | Notes |
|---|---|---|
| **mimicgen** (`eef_delta`, Panda 2-view) | `panda`, and *geometry-compatible* 7-DoF arms (`fr3`, `fr3_v2`, `kinova_gen3`, `sawyer`, `kuka_iiwa`, `ur5e`/`ur10e`) | Trained on **Panda** in MimicGen sim. Other arms only work if the IDM's Jacobian + action denorm match that arm's kinematics — **eef-delta is the most transferable** (Cartesian, not joint-space), but still needs per-arm validation. |
| **droid** (`cartesian_delta`, FR3) | `fr3`, `fr3_v2` (DROID is FR3) | Wave-2; cartesian_delta again relatively transferable across 6-DoF arms. |
| **allegro** (`joint_position`, 16-DoF hand) | `allegro_hand` | Wave-2; joint_position is embodiment-EXACT — only the Allegro hand. |
| **pusht** (`pos`, planar) | *(none — a 2D sim task, not a registry robot)* | PushT is a planar pusher sim, not one of the 68. Use it for smoke-testing the pipeline only. |

**Categories VERA can NEVER drive** (no trained planner/IDM, wrong action space):
`humanoid` (18), `mobile`/quadruped (10), `aerial` (2), `bimanual` (3),
`mobile_manip` (4), `expressive` (1), most `hand` (7 of 8) → **38 robots out of
scope by model availability**, and the remaining `arm`/`hand` need an IDM trained
(or transfer-validated) for that specific embodiment.

> This is the SAME limitation every VLA provider has: `groot`, `lerobot_local`,
> `cosmos3` each only work for the embodiments their checkpoints were trained on.
> VERA is not special here. "Works with all 68" is not achievable with 4 models.

---

## 3. 🔴 The REAL bug — action columns are NOT bound to robot actuators

Even for the supported embodiments, the provider currently **cannot drive a
strands-robots robot**, because of how it names action columns.

### The chain (verified)
1. `SimEngine` calls `policy.set_robot_state_keys(self.robot_joint_names(robot))`
   before rollout (`simulation/base.py:445`, `mujoco/simulation.py:2306`).
2. `PolicyRunner` calls `policy.get_actions(...)` → `action_dict` →
   `sim.send_action(action_dict, robot_name=...)` (`policy_runner.py:472,487`).
3. `send_action` matches `action_dict` **keys** to the robot's actuator/joint
   names; unmatched keys go into `self._unresolved_action_keys` and are
   **silently dropped** (`mujoco/simulation.py:~291-340`).

### The bug
`VeraPolicy._action_column_names` (provider.py:289) returns
``["action_0", "action_1", …]`` and ONLY renames them if an explicit
``action_mapping`` was hand-passed. It **stores** ``set_robot_state_keys`` into
``self._robot_state_keys`` (line 177) but **never uses it**. So a real robot
(e.g. `so100` joints `shoulder_pan, shoulder_lift, …`) receives action keys
`action_0..action_5` → **every key is unresolved → the robot does not move.**

### Proof of the gap
- `cosmos3` (the provider VERA was modelled on) resolves columns from a real
  **action layout** and falls back to `action_{i}` only for overflow
  (`cosmos3/policy.py:488-512`), AND uses `robot_state_keys`. VERA dropped that.
- `VeraPolicy` grep: ``_robot_state_keys`` appears at lines 152, 176, 177 only —
  **storage, never read.**

---

## 4. ✅ The fix — bind VERA's action columns to robot actuators

Make `_action_column_names` use, in priority order:
1. explicit ``action_mapping`` (caller override) — keep;
2. the server's **action_space contract** to lay out columns sensibly, then map
   onto ``self._robot_state_keys`` (the actual robot joints) — NEW;
3. ``action_{i}`` fallback only when nothing else resolves — keep.

Concretely, per VERA `action_space`:

- **`joint_position`** (allegro): action dims == robot joints, **positional** →
  map column ``i`` → ``self._robot_state_keys[i]`` directly (after sanity-checking
  ``len`` match). This is exactly cosmos3's joint path.
- **`eef_delta` / `cartesian_delta`** (mimicgen / droid): the chunk is a 6-DoF
  **end-effector delta + gripper**, NOT per-joint targets. These CANNOT be written
  to joint actuators directly — they need either (a) the sim's Cartesian/IK action
  interface, or (b) an IK step to convert eef-delta → joint targets keyed by
  ``self._robot_state_keys``. cosmos3 has ``sim_ik.py`` for precisely this; VERA
  should reuse the same IK seam (see `policies/cosmos3/sim_ik.py`). Without it,
  eef-delta embodiments produce keys the sim can't apply.
- **`pos`** (pusht): 2D planar — a sim-task action, map to the pusht env's 2 dims;
  not a registry robot.

### Minimal correct change
1. In `_action_column_names`, when ``action_mapping`` is absent AND
   ``action_space`` is joint-positional AND ``len(self._robot_state_keys) ==
   action_dim`` → return ``list(self._robot_state_keys)`` (optionally keeping a
   gripper column name from the robot's gripper joint).
2. For eef/cartesian-delta embodiments, route through an IK adapter
   (reuse/share ``cosmos3/sim_ik.py``) to emit joint-keyed targets; do NOT emit
   raw ``action_i`` deltas that ``send_action`` will drop.
3. Add a loud one-time warning if columns resolve to ``action_i`` (i.e. nothing
   bound) — mirrors cosmos3's "set robot_state_keys" guidance so failures are
   visible, not silent.

---

## 5. Coverage matrix (after the fix)

| Category | # | VERA-drivable? | Why |
|---|---|---|---|
| arm | 22 | **partial** | mimicgen(Panda)/droid(FR3) eef-delta + IK; per-arm validation needed; joint-space arms need a matching IDM |
| hand | 8 | **1** (`allegro_hand`) | allegro joint_position; Wave-2 ckpt pending |
| humanoid | 18 | **no** | no whole-body/locomotion planner, wrong action space |
| mobile / quadruped | 10 | **no** | locomotion, not manipulation |
| mobile_manip | 4 | **no\*** | arm sub-chain *might* take eef-delta, but base+arm coordination is out of scope |
| bimanual | 3 | **no** | no bimanual VERA planner |
| aerial | 2 | **no** | flight control, not VERA |
| expressive | 1 | **no** | not a manipulation policy |

**Realistic end-to-end today:** PushT (smoke) + MimicGen→Panda (and IK-validated
6/7-DoF arms). Everything else is gated on (a) a VERA checkpoint existing for that
embodiment, and (b) the action-binding/IK fix above.

---

## 6. Recommendations (in priority order)

1. **Fix the action-binding bug** (§4) — without it, even Panda doesn't move.
   This is the only true *code* defect; everything else is model availability.
2. **Add an IK adapter for eef/cartesian-delta** by sharing `cosmos3/sim_ik.py`
   so mimicgen/droid can drive any kinematically-compatible arm.
3. **Document the embodiment contract** in the provider docstring: VERA serves
   ITS 4 embodiments; a robot is drivable iff (matching action_space) AND
   (a checkpoint exists) AND (IK/validation done). Don't imply universal support.
4. **Add a guard** in `VeraPolicy.__init__` / `reset`: if the server's
   ``action_space`` is eef/cartesian-delta and no IK adapter / action_mapping is
   configured, warn that joint binding is required.
5. **Tests:** a per-action_space binding test (joint_position → exact joint keys;
   eef_delta → IK path emits joint keys; assert zero ``unresolved_keys`` against a
   real registry robot's ``robot_joint_names``).

---

## TL;DR
- VERA = 4 trained embodiments (2 with ckpts today). It is a manipulation policy;
  ~38 of our 68 robots are out of scope by category, the rest need a matching
  trained IDM. **"All embodiments" is not achievable with 4 models** — same as
  every other VLA provider.
- **But there's a genuine bug:** the provider emits ``action_0..N`` and never binds
  to the robot's real actuator names (it stores ``robot_state_keys`` but never uses
  them), so ``send_action`` drops every command. cosmos3 does this correctly;
  VERA must too — joint_position via direct key map, eef/cartesian-delta via a
  shared IK adapter. Fix that and mimicgen→Panda (+ IK-validated arms) works.
