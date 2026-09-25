---
description: NVIDIA GR00T Whole-Body-Control (SONIC) humanoid locomotion - in-process ONNX, no GPU required, goal via target_velocity kwargs.
---

# WBC (Whole-Body-Control)

[`WBCPolicy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/wbc/policy.py)
wraps NVIDIA's
[GR00T Whole-Body-Control](https://github.com/NVlabs/GR00T-WholeBodyControl)
(SONIC / decoupled-WBC) ONNX controllers for deploy-grade humanoid locomotion
on the Unitree G1. It runs **in the same process** through ONNX Runtime on CPU:
no sidecar, no network round-trip, no GPU.

It is a non-VLA locomotion controller: it reads its goal from the well-known
locomotion `**kwargs` (`target_velocity`), ignores camera frames
(`requires_images = False`), and never parses the instruction string. It drives
the **15 leg+waist DOFs** of the G1 and holds the arms at their nominal defaults;
to drive the arms as well, layer a manipulation policy on top with
[`CompositePolicy`](wbc-rollouts.md#composing-an-upper-body-manipulation-on-top-of-wbc).

## Install

```bash
pip install "strands-robots[wbc]"            # onnxruntime only - light, no torch
pip install "strands-robots[wbc,sim-mujoco]" # + MuJoCo to drive the G1 in sim
```

No weights are bundled and there is no default download: a bare
`create_policy("wbc")` raises instead of fetching the wrong model family. The
two decoupled-WBC G1 controllers are 1.8 MB each inside the 4.6 GB git-LFS tree
of [`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl).
`media.githubusercontent.com` serves the LFS content of a public repository, so
fetch the two files (3.6 MB of ONNX) instead of cloning everything around them:

```bash
mkdir -p /path/to/grootwbc-g1 && cd /path/to/grootwbc-g1
url=https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main/decoupled_wbc/sim2mujoco/resources/robots/g1/policy
curl -LO "$url/GR00T-WholeBodyControl-Balance.onnx"   # main (Balance) policy
curl -LO "$url/GR00T-WholeBodyControl-Walk.onnx"      # optional walk policy
```

The canonical `GR00T-WholeBodyControl-Balance.onnx` / `-Walk.onnx` filenames are
accepted verbatim; `policy.onnx` / `walk_policy.onnx` also work and take
precedence when both are present:

```text
/path/to/grootwbc-g1/
    GR00T-WholeBodyControl-Balance.onnx   # main (Balance) policy
    GR00T-WholeBodyControl-Walk.onnx      # optional walk policy
    config.json                           # optional
```

The HuggingFace repo [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC)
is the SONIC VLA inference stack, **not** this Balance/Walk family; passing it as
a checkpoint raises.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "wbc",                                  # shorthand: "sonic"
    checkpoint="/path/to/grootwbc-g1",       # dir with policy.onnx (+ walk_policy.onnx)
    walk=True,
)

actions = policy.get_actions_sync(
    observation_dict={"observation.state": [0.0] * 29},  # G1 joint positions
    instruction="walk forward",             # ignored by the controller
    target_velocity=[0.5, 0.0, 0.0],        # [vx, vy, omega] (m/s, m/s, rad/s)
)
# actions == [{"left_hip_pitch_joint": .., ..., "waist_pitch_joint": ..}]
# one per-tick dict of 15 leg+waist joint targets (closed-loop, not a chunk)
```

`target_velocity` is held to a locomotion envelope of ±2.0 m/s per linear
component and ±2.0 rad/s for `omega`, at the policy and again on the mesh
before dispatch (one definition, `strands_robots.locomotion_envelope`). A
component past it is refused with a reason, never clamped. A faster platform
raises the bound with `STRANDS_MAX_TARGET_LINEAR_VELOCITY_MPS` /
`STRANDS_MAX_TARGET_ANGULAR_VELOCITY_RPS` (positive floats, read on every
call).

## Parameters

```python
WBCPolicy(
    checkpoint="/path/to/grootwbc-g1",  # dir with policy.onnx, a direct .onnx path, or an HF id
    config=None,                       # WBCConfig | path | dict | None (None -> config.json in checkpoint)
    walk=True,                         # load + prefer walk_policy.onnx for locomotion
    target_velocity=None,              # constructor-time default [vx, vy, omega] (per-call kwarg overrides)
    allow_missing_models=False,        # test seam: skip eager ONNX load (inject a stub session)
)
```

A missing `onnxruntime` or a missing checkpoint raises `RuntimeError` at
construction - WBC never falls back to silent zero torques.

`walk` and `allow_missing_models` each select a posture, so both are checked
rather than read by truthiness: a non-boolean raises `ValueError` naming the
parameter. A string such as `"false"` - the spelling a JSON `policy_config`
reaches for - is truthy, so it is refused by name rather than read as the test
seam.

### Config value domain

`WBCConfig` refuses an unusable *value* at construction, because every numeric
field is read verbatim into the PD law that writes `data.ctrl` or into the
observation, so an unusable one becomes a wrong torque rather than an error:

| Field | Accepted | Why |
|-------|----------|-----|
| `action_scale` | finite `> 0` | The only path from the network to the joint targets. `0` (or `False`) makes `target_q == default_angles`, discarding the policy; a negative value inverts every offset. |
| `kps`, `kds` | finite `>= 0`, per component | `kp = 0` with `kd > 0` is a pure-damping joint and stays valid; a *negative* gain makes `(target_q - q) * kp` drive the joint away from its target. |
| `default_angles`, `cmd_scale`, `rpy_cmd` | finite, per component | Signed quantities (a stance angle, a yaw rate, a roll target), so only finiteness is constrained. |
| `cmd_scale`, `obs_scales` (arity) | stated, or empty to mean the upstream default | An EMPTY `cmd_scale` means "not stated" and is completed with `(2.0, 2.0, 0.5)` at construction, so it cannot scale the velocity differently from omitting the argument. A wrong NON-empty length is still refused by name. |
| `obs_scales` values, `height_cmd`, `freq_cmd` | finite | A non-finite scale poisons the observation frame the network is given. |

A `nan`/`inf` or a `None` anywhere surfaces as a `ValueError` naming the field
and component at construction, not as a non-finite torque mid-rollout.

## Goal kwargs

WBC reads locomotion commands from `**kwargs`, sharing the non-VLA goal
vocabulary so a command can flow through `run_policy` / mesh `tell()` without
coupling to a backend:

| Key | Type | Accepted | Meaning |
|-----|------|----------|---------|
| `target_velocity` | `list[float]` | numeric, >= 3 entries, every component finite | Locomotion command `[vx, vy, omega]` (m/s, m/s, rad/s). Scaled by `cmd_scale` (`[2.0, 2.0, 0.5]`) into the observation's command block. |
| `target_orientation` | `list[float]` | numeric, >= 3 entries, every component finite | Target base `[roll, pitch, yaw]` (rad), written to command slots `[4:7]`. Defaults to the config `rpy_cmd` (`[0,0,0]`). |
| `height` | `float` | finite | Target base height (m), written to command slot `[3]`. Defaults to the config `height_cmd` (`0.74`). |

A per-call `target_velocity` overrides the constructor-time default; with no
command the controller holds a standing balance, and `None` means "not
supplied". `target_velocity` is one of the issue #300 well-known goal keys, so
the mesh path forwards it the same way it forwards a planner's goal:
`mesh.tell(peer, "walk forward", policy_provider="wbc", target_velocity=[0.5, 0.0, 0.0])`.
`target_orientation` and `height` are WBC's own kwargs. Each key keeps the
domain `WBCConfig` enforces for the field it overrides; the vector keys need at
least three components because the command block is zero-initialised, so a
shorter one would silently command zero for an unmentioned axis (a longer one is
truncated).

## Control contract

WBC reproduces the upstream `GearWbcController` loop (NVlabs/GR00T-WholeBodyControl
`decoupled_wbc/sim2mujoco`, `run_mujoco_gear_wbc.py` + `g1_gear_wbc.yaml`):

- **Two ONNX sessions** - `policy.onnx` and an optional `walk_policy.onnx`,
  loaded once. When the **raw** velocity-command norm is `<= 0.05` the main
  (standing) policy runs; above that the walk policy (when `walk=True`).
- **Observation** - an 86-dim frame stacked over `obs_history_len` (default 6,
  network input `86 * 6 = 516`): command `[0:7]` =
  `[vx*2.0, vy*2.0, omega*0.5, height, roll, pitch, yaw]`, base angular velocity
  `[7:10]` (`ang_vel_scale=0.5`), projected gravity `[10:13]`, joint positions
  `[13:28]` (minus `default_angles`, `dof_pos_scale`), joint velocities
  `[28:43]` (`dof_vel_scale=0.05`), previous action `[43:58]`, reserved zero tail
  `[58:86]`. The upstream YAML's flat scale keys are normalised into the nested
  `obs_scales` map, and a config states only the scales it changes; the rest keep
  the upstream defaults. An empty `cmd_scale` means "not stated" and resolves to
  `(2.0, 2.0, 0.5)`, never to a bare `1.0` - which would command a yaw rate
  double the one asked for.
- **Action** - a 15-dim joint-position *offset*; the policy forms
  `target_q = default_angles + action_scale * raw`, keyed by actuator name. For
  torque-actuated MuJoCo, convert with `policy.compute_torques(target, q, dq)`.

## Actuator mapping

WBC output index `i` drives `WBC_G1_LEG_WAIST_JOINTS[i]` - an explicit table.
`set_robot_state_keys` validates that the robot's first 15 joints match this
order and raises otherwise:

```
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint,
waist_yaw_joint, waist_roll_joint, waist_pitch_joint
```

## Rollouts

Running WBC - the torque shim on a position-servo scene, a camera that
follows the walk, the torque-deploy loop and a composite upper body - is on
[WBC rollouts](wbc-rollouts.md).

## Gait-clock variant

NVIDIA's reference repo ships a second G1 controller - a single-policy
**gait-clock** variant (95-dim observation, an 8-wide command with a
`freq_cmd` step-frequency slot, and a 2-dim bipedal phase clock). It is
implemented by `WBCGaitPolicy` (provider `wbc_gait`). See
[WBC gait-clock variant](wbc_gait.md).

## See also

- [Policy overview](overview.md)
- [GR00T](groot.md) - ZMQ service VLA (manipulation upper body).
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
