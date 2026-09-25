---
description: lerobot_local observation binding - normalization stats, unit frames, state keys and camera routing.
---

# LeRobot Local: observation binding

How a runtime observation reaches a LeRobot checkpoint under the
[`lerobot_local`](lerobot-local.md) provider: normalization stats and their
units, the joint keys that compose `observation.state`, and the camera routed
onto each declared image feature.

## Processor bridge and normalization

`use_processor=True` (default) wraps the policy in the checkpoint's
pre/post-processor pipelines, so observations are normalized going in and
actions come back in physical joint units. The bridge takes them from
`policy_preprocessor.json` / `policy_postprocessor.json` when the checkpoint
ships them; a checkpoint that instead carries `normalize_inputs.*` /
`unnormalize_outputs.*` buffers in `model.safetensors` (zoo checkpoints such as
`lerobot/act_aloha_sim_transfer_cube_human`) has both pipelines rebuilt from
those buffers with lerobot's `extract_normalization_stats` +
`make_pre_post_processors`. MolmoAct2 checkpoints take neither path: lerobot's
own factory reads their `norm_stats.json`, and a `norm_tag` the file does not
declare is refused by lerobot naming the tags it does (see
[MolmoAct2](lerobot-local.md#molmoact2)). `processor_overrides` is keyed by step
name (a step's `registry_name`, or its class name when it is not
registry-backed) and each override is routed to the pipeline that owns that step:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/smolvla_base",
    policy_type="smolvla",
    processor_overrides={
        "normalizer_processor": {"stats": dataset_stats},    # observation.state
        "unnormalizer_processor": {"stats": dataset_stats},  # action
    },
)
```

Naming only one leaves the other inert, and the diagnostic keeps reporting
whichever half is still unnormalized. Fine-tuning writes stats under the
canonical keys and needs no override.

A pretraining base checkpoint carries its training datasets' stats under prefixed
keys rather than the canonical ones LeRobot looks up, so the diagnostic lists the
spellings it found per missing key -- `lerobot/smolvla_base` reports
`{'observation.state': [], 'action': ['so100-blue.buffer.action',
'so100-red.buffer.action', 'so100.buffer.action']}`. Remap one onto `action` and
pass it as the override above. They are listed rather than adopted because the
three describe different distributions, so the choice is the caller's; an empty
list means the checkpoint ships no stats for that feature.

Stats also carry the *units* the dataset was recorded in, and that is the second
half a sim caller owes. An SO-arm dataset comes through the driver's
`MotorNormMode` - arm joints in servo **degrees**, gripper in `RANGE_0_100`
(`smolvla_base`'s `so100.buffer.action.std` is
`[26.4, 52.4, 49.9, 37.0, 59.4, 19.0]`) - while a MuJoCo state is **radians**.
Feeding radians to degree stats is not a small error, it is a change of scale, so
`observation.state` reaches the model as a near-constant:

| `state_units` | full so101 joint range, in sigma |
| --- | --- |
| `"native"` (radians) | 0.07 -- 0.15 |
| `"degrees"` | 3.8 -- 8.3 |

Declare both halves together, stats and units:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/smolvla_base",
    policy_type="smolvla",
    embodiment="so101",                                      # units: rad -> deg, gripper -> 0..100
    processor_overrides={
        "normalizer_processor": {"stats": dataset_stats},    # observation.state
        "unnormalizer_processor": {"stats": dataset_stats},  # action
    },
)
```

Together, because the units half on its own is refused rather than run: a map
that converts writes its conversion into the very tensor the inert normalizer
then leaves alone. The load names the inert features and both ways out - supply
the stats above, or drop the conversion (`set_robot_state_keys([...])`, or a
`"native"` map).

The built-in `so100` / `so101` maps declare `state_units`/`action_units`
`"degrees"`; every other map defaults to `"native"`, which is right for real
hardware - an SO follower already reports driver units - and wrong for a sim
packing radians. Those two spellings are the whole vocabulary
(`embodiment.UNIT_FRAMES`); any other is refused wherever a frame is held - when
the map is built, and when LeRobot rebuilds the pack-state step from a saved
`policy_preprocessor.json`, where `"DEGREES"` (LeRobot's own `MotorNormMode`
spelling) would otherwise mean `"native"` and convert nothing.

Both halves of a declared map are installed as *preprocessor* steps, so a
checkpoint that ships no `policy_preprocessor.json` (only a postprocessor) has
nothing for them to be installed into. Declaring an `embodiment` against one is
refused at load, naming the missing pipeline, because the alternative is half a
map. Drop `embodiment=` to use the raw obs/action flow, or load a checkpoint that
ships a preprocessor. `joint_mids` is the companion knob:
LeRobot's `DEGREES` mode is mid-point centered, so without it sim `qpos=0` is
taken to be the calibration mid.

Supplied stats must be as wide as the features the checkpoint declares - a
6-DOF SO-101, a 7-DOF arm and a 14-DOF bimanual all have `observation.state`.
A mismatch is refused at load,
naming the feature and both widths, rather than surfacing as a tensor broadcast
error on the first inference. Visual `(C,)` stats are exempt (LeRobot reshapes
them to `(C, 1, 1)`):

```
ValueError: lerobot_local: lerobot/smolvla_base was given normalization stats
that do not match the widths the checkpoint declares: ["observation.state
(STATE/MEAN_STD): feature declares width 6, stats 'observation.state.mean'
supply 7", ...]
```

## State routing

`observation.state` is composed from `robot_state_keys`, set with
`set_robot_state_keys([...])` or by an `embodiment`. When only some keys are
present the vector is partly bound and the policy reports the absent keys, the
keys the observation does carry and the remedy; when none are present it falls
back to the observation's own state vector. Both degradations are logged, and
`strict_keys=True` turns them into raises.

The remedy names an `embodiment=` only when a shipped one declares `state_keys`
the observation carries, so following it cannot land back on the same mismatch.
One exception is reported instead of recommended. A declared embodiment that was
already **rejected** at load time (its `obs_rename` names an image feature the
checkpoint does not declare, so the whole map including the state binding is
discarded - see [the pre-flight check](#embodiment-obs_rename-and-the-pre-flight-check))
would loop if re-passed, so the remedy says it was rejected and points at
`camera_key_map=` / `obs_rename_override=` to make it validate, or
`set_robot_state_keys([...])`.

A candidate that converts units is withheld too when normalization is inert
(the "stats do not cover" warning above): `so100` and `so101` declare
`state_units='degrees'`, correct only against degree-recorded stats, and with
none the so101 joint range reaches the model at up to 160.0 where packing it
natively reaches 2.79. The remedy points at `set_robot_state_keys([...])`, which
leaves the units alone, and names the `processor_overrides` that would make the
embodiment correct.

## Camera routing

Observations use bare camera names (`top`, `wrist`); the policy declares image
inputs as `observation.images.*`. Each camera is routed by, in order:

1. `camera_key_map={"front": "observation.images.top", ...}` when given;
2. the embodiment's `obs_rename` map (below);
3. a name heuristic (exact stem, then substring).

### Embodiment `obs_rename` and the pre-flight check

`embodiment="so101"` routes cameras from the embodiment's `obs_rename`
(`{runtime_camera_name: "observation.images.*"}`), under `camera_key_map` and
then `obs_rename_override`. A `camera_key_map` entry replaces the source key the
embodiment declares for the feature it claims, so a scene whose cameras are
named for the scene routes onto an embodiment without renaming them:

```python
# so101 declares front -> .../image and wrist -> .../wrist_image
create_policy("lerobot_local", pretrained_name_or_path=..., embodiment="so101",
              camera_key_map={"cam_top": "observation.images.image",
                              "cam_wrist": "observation.images.wrist_image"})
# routed obs_rename: {cam_top: .../image, cam_wrist: .../wrist_image}
```

`obs_rename_override` is applied last, because a falsy value there is the only
way to DROP a declared rename (`{"wrist": None}` adapts a two-camera embodiment
to a single-camera checkpoint). An entry in either map naming an image feature
the model does not declare is refused by name.

Before the model is built the policy checks that every declared image feature
has a source in the runtime observation - after routing both maps, so a camera
you bound explicitly satisfies it - and refuses otherwise, naming both sides:

```text
Embodiment 'so101' cannot route cameras to the model's image feature(s)
['observation.images.image', 'observation.images.wrist_image']: none of the
expected source key(s) ['front', 'wrist'] are in the runtime observation, which
provides [...]. Either: (a) rename your sim cameras to one of ['front', 'wrist']
..., or (b) pass policy_config={'camera_key_map': {...}} ...
```

### When the checkpoint declares different image features

An embodiment's rename targets are a *guess* about a checkpoint's feature names.
`so101` feeds `observation.images.image` + `.../wrist_image`;
`lerobot/smolvla_base` declares `observation.images.camera1..3`, and no camera
name can satisfy a target the model does not declare. The pre-flight check reads
the checkpoint's declared features (from its `config.json`, before the weight
download) and reports that mismatch instead of asking for a rename that cannot
help:

```text
Embodiment 'so101' feeds image feature(s) ['observation.images.image',
'observation.images.wrist_image'], which 'lerobot/smolvla_base' does not declare
- it declares ['observation.images.camera1', 'observation.images.camera2',
'observation.images.camera3']. ... Route the features it does declare instead:
policy_config={'obs_rename_override': {'front': None, 'wrist': None,
'camera1': 'observation.images.camera1', ...}} - a falsy value drops a rename
this checkpoint cannot accept.
```

Both halves are needed: the drops alone leave the declarative path with no
camera routing, and the model then raises "All image features are missing from
the batch".

A single-camera checkpoint needs no embodiment: declare the joint names with
`set_robot_state_keys([...])` and the policy synthesizes a state-only embodiment
that routes the one declared image feature to the one camera.

## See also

- [LeRobot Local](lerobot-local.md) - install, parameters, model caching, RTC
- [Camera naming](camera-naming.md) - the camera key each embodiment declares
- [MolmoAct2](molmoact2.md) - action contract, units, and motion diagnostics
