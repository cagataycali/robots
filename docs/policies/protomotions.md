# ProtoMotions — reference-motion tracking for the Unitree G1

`ProtoMotionsPolicy` wraps a ProtoMotions **Generalist Tracking Policy** (GTP)
ONNX export. Given a reference motion clip it emits balanced PD joint targets
for the Unitree G1's 29 actuators, tracking that clip while keeping the robot
upright. It is the tracking half of a two-stage pipeline: a *kinematic*
generator such as [`KimodoPolicy`](./kimodo.md) produces a `qpos` sequence with
no notion of balance, and the tracker turns it into physics. Compare
[`WBC`](./wbc.md), which takes a velocity/height *command* rather than a
reference clip.

## Install

```bash
pip install "strands-robots[protomotions]"
```

That pulls `onnxruntime` (runs the graph), `pyyaml` (reads the
`unified_pipeline.yaml` sidecar) and `huggingface_hub` — which you call yourself
to fetch the checkpoint, because `onnx_path` and `yaml_path` take local files and
this policy resolves no model id. Weights are not bundled. Building a reference clip from `qpos` with
[`qpos_to_motion_data`](#bridging-a-qpos-clip) additionally needs MuJoCo, which
ships in `strands-robots[sim-mujoco]`.

## Run a clip in simulation

```python
from strands_robots import Robot
from strands_robots.policies.protomotions import (
    ProtoMotionsPolicy,
    qpos_to_motion_data,
)

sim = Robot("unitree_g1", mode="sim")

motion = qpos_to_motion_data(qpos, fps=30, proto_mjcf_path=mjcf)
policy = ProtoMotionsPolicy(
    onnx_path="unified_pipeline.onnx",
    yaml_path="unified_pipeline.yaml",
    motion=motion,
)

sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    n_steps=1000,
    control_frequency=50.0,
)
```

## The observation contract

The tracker consumes four inputs per tick. Three are ordinary proprioception the
runtime already publishes: joint positions (`<joint>`), joint velocities
(`<joint>.vel`) and the base angular velocity (`base_ang_vel`, the freejoint's
`qvel[3:6]`, already in the body frame). The fourth is the **world orientation
of the anchor link**, `torso_link` on the G1 - not derivable from `base_quat`,
which is the *pelvis*: with the waist swept through 0.6 rad the two frames
diverge by up to 42 degrees. So the policy declares the link it needs, and the
runtime resolves it once per rollout and merges `body.torso_link.quat` into every
observation (the same "policy declares, runtime supplies" contract as
[`requires_images`](./overview.md)):

```python
policy.required_bodies        # ('torso_link',)
```

A caller assembling observations by hand (a hardware loop reading an IMU) passes
the two signals directly, or puts them on the observation dict under either
spelling:

```python
await policy.get_actions(
    obs,
    "",
    anchor_rot_xyzw=[x, y, z, w],   # anchor link, world frame
    root_ang_vel_local=[wx, wy, wz],
)
```

| signal | accepted observation keys |
| --- | --- |
| anchor rotation, `xyzw` | `anchor_rot_xyzw`, `observation.anchor_rot_xyzw` |
| root angular velocity, local frame | `root_ang_vel_local`, `observation.root_ang_vel_local` |

Missing signals are refused, not substituted: no anchor pose raises naming the
key it wanted (`base_quat` stands in only when the config's anchor body *is* the
floating base), and an absent `<joint>.vel` is refused rather than read as zero.

Orientations need not be exactly unit: both rotation helpers normalise their
input, so a drifted IMU reading or a lerped sample (up to 8% short, which read
as-is is a heading 6.2 degrees off) gives the unit quaternion's answer:

```python
from strands_robots.policies.protomotions import extract_yaw_quat

extract_yaw_quat(q)          # same heading as
extract_yaw_quat(q * 0.92)   # this
```

An orientation that cannot define a rotation - all zeros, or a non-finite
component - is refused with a `ValueError` naming the helper and the value.

## Bridging a qpos clip

`qpos_to_motion_data` turns a `[T, 7 + 29]` MuJoCo `qpos` sequence into the
cache the tracker plays: forward kinematics through the ProtoMotions MJCF for
per-body world poses, finite-differenced velocities at the source rate,
resampled onto the tracker's `control_dt` (0.02 s):

```python
cache = qpos_to_motion_data(qpos, fps=30, proto_mjcf_path=mjcf)
cache["num_frames"], cache["control_dt"]
```

`MotionPlayer` accepts that dict, an `.npz` from `MotionPlayer.save_cache_npz`,
or a raw ProtoMotions `.pt`. Four things to know about the cache:

- **Frame counts must agree.** Every channel is `[num_frames, ...]`; trimming
  the channels and leaving `num_frames` behind is refused with both counts
  named. Drop `num_frames` (or set it) after editing:

```python
cache["dof_pos"] = cache["dof_pos"][:100]   # ... and the other five channels
del cache["num_frames"]                     # or set it to 100
player = MotionPlayer(cache)
```

- **The MJCF has to be the tracker's own embodiment.** The tracker reads bodies
  by **row index** into `GTP_G1_BODY_NAMES` (33 names from the checkpoint's
  sidecar), so `proto_mjcf_path` must carry all 33 bodies plus a free root and
  the 29 `GTP_G1_JOINT_NAMES` joints (`qpos` width 36). The common fingerless
  G1 models expose 30 bodies (no `head`, no `rubber_hand`s) and the hand
  variants 44; both are refused naming what is missing, since a positional
  read of a 30-body model hands the tracker the wrong link for `torso_link`.
- **A floor is added only when the model has none.** The bridge appends a
  plane geom named `floor` unless MuJoCo's parsed geom list already has a
  ground (a `unitree_ros` second `<worldbody>`, a menagerie `scene.xml`
  `<include>`), in which case the file is used unchanged.
- **Cache velocities are world-frame.** `body_vel` / `body_ang_vel` follow the
  ProtoMotions motion-library convention; `compute_root_local_ang_vel` rotates
  them into the root frame, so a hand-built cache holding local-frame rows is
  rotated twice - whole rad/s off on a walking clip.
- **Motion files are read with a restricted unpickler.** `.npz` needs no torch;
  a `.pt` is read with `torch.load(..., weights_only=True)`, which accepts
  tensors and scalars only, because clips travel and an unrestricted unpickler
  executes what the file names. A refused `.pt`: re-save it as a dict of
  tensors, or convert once with `save_cache_npz`.

## Per-episode reset

`reset(seed=...)` rewinds the playhead and clears the historical-action buffer.
The runtime calls it once per episode, so a multi-episode `eval_policy` run
replays the clip from frame 0 each time. The seed is accepted for interface
parity and ignored: given a reference clip and an observation the tracker is
deterministic, holding no RNG state.

## Configuration

`ProtoMotionsConfig` is a frozen dataclass mirroring the checkpoint's
`unified_pipeline.yaml`: the 29 joint names in ONNX action order, the 33 body
names, the anchor and root body indices, per-joint PD gains, timing, and the
lookahead offsets for the future-reference window. Pass `yaml_path=` to load a
sidecar; omit it to use the defaults, which match the shipped export.

| Field | Default | Meaning | Accepted |
| --- | --- | --- | --- |
| `joint_names` | 29 G1 joints | ONNX action order | |
| `anchor_body_index` | `16` (`torso_link`) | Link whose world rotation the network reads | a whole number in `0..len(body_names)-1`; negative indices and booleans are refused, an integral float is kept as its row |
| `root_body_index` | `0` (`pelvis`) | Floating base | same |
| `control_dt` | `0.02` | Seconds per control tick (50 Hz); the period the reference motion is resampled onto | finite, `> 0`; a cache dict's own `control_dt` outranks the argument and is held to the same domain by `MotionPlayer` |
| `future_step_indices` | `(1, 2, 4, 8)` | Lookahead offsets, in control steps | |
| `action_ema_alpha` | `1.0` | Smoothing weight on the emitted joint targets; `1.0` is passthrough | finite, in `(0, 1]` |

Every domain is checked when the config is built, from a sidecar or by hand, not
when a field is read. The values turned away are not near-misses: a negative or
infinite `control_dt` collapses a 3-second clip to one frame for the whole
episode, `true` (read as `1.0`) plays it in three ticks, `action_ema_alpha=0`
freezes the first target and `nan` poisons every later tick. `physics_dt` and
`decimation` are unchecked: nothing in this package reads them.

### Target smoothing

`action_ema_alpha` is the weight the current network output carries in the
target the PD loop receives, `y[t] = alpha * x[t] + (1 - alpha) * y[t-1]`.
`1.0` returns the output unchanged and bit-exact; a smaller value trades
tracking lag for less per-tick jitter. Measured on an output with an alternating
+/-0.11 rad component, the mean per-tick change in `left_hip_pitch_joint`:

| `action_ema_alpha` | mean per-tick change |
| --- | --- |
| `1.0` (passthrough) | 0.220 rad |
| `0.5` | 0.074 rad |
| `0.2` | 0.029 rad |
| `0.05` | 0.010 rad |

The first tick seeds the filter from the network's own output, not zeros (a
zero seed would lurch a standing humanoid toward the zero pose), and the
historical-actions buffer keeps the RAW output, since the graph's
`historical_processed_actions` input is defined over it.

## Testing without weights

`session=` injects anything satisfying the `ProtoMotionsSession` protocol
(`run(output_names, inputs) -> list[np.ndarray]`), so the observation to action
mapping can be exercised with no onnxruntime, no weights and no GPU:

```python
policy = ProtoMotionsPolicy(session=stub, motion=cache)
```

Outputs are paired to the names in `config.onnx_out_names` rather than read
positionally, so an export that declares them in another order still feeds
`joint_pos_targets` to the PD loop; an export missing that output is refused by
name.
