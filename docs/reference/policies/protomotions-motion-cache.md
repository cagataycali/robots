---
description: Turning a MuJoCo qpos sequence into the motion cache the ProtoMotions tracker plays.
---

# Bridging a qpos clip

How a kinematic `qpos` sequence becomes the cache a
[`ProtoMotionsPolicy`](protomotions.md) plays: what the six channels hold, which
MuJoCo model may produce them, and the file formats `MotionPlayer` accepts. The
tracker itself - its install line, its observation contract, its config fields -
is [ProtoMotions](protomotions.md).

`qpos_to_motion_data` turns a `[T, 7 + 29]` MuJoCo `qpos` sequence into the
cache the tracker plays: forward kinematics through the ProtoMotions MJCF for
per-body world poses, finite-differenced velocities at the source rate,
resampled onto the tracker's `control_dt` (0.02 s):

```python
cache = qpos_to_motion_data(qpos, fps=30, proto_mjcf_path=mjcf)
cache["num_frames"], cache["control_dt"]
```

`MotionPlayer` accepts that dict, an `.npz` from `MotionPlayer.save_cache_npz`,
or a raw ProtoMotions `.pt`. Things to know about the cache:

- **The six channels are the required content.** `control_dt` and `num_frames`
  are optional on every route: omit `control_dt` and the `control_dt=` argument
  stands, omit `num_frames` and the channels' own row count is used. A cache
  short of a channel is refused by naming every channel it lacks - and, when it
  came from a file, the file.
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
  ProtoMotions' own G1 MJCFs are that embodiment, and no asset is bundled here,
  so fetch one. Which one matters for the download, not for the cache:
  `g1_bm.xml`, `g1_bm_box_feet.xml`, `g1_bm_no_mesh_box_feet.xml` and
  `g1_holo_compat.xml` produce byte-identical `body_pos`, while `g1_holo.xml` is
  the fingerless case above and is refused by name. Only the no-mesh variant
  loads from a single file:

```bash
curl -LO https://raw.githubusercontent.com/NVlabs/ProtoMotions/main/protomotions/data/assets/mjcf/g1_bm_no_mesh_box_feet.xml
```

  The other three reference `../mesh/G1/*.stl`, which upstream keeps in Git LFS:
  a `raw.githubusercontent.com` copy of those files is a pointer, and MuJoCo
  refuses it with `decoder failed for mesh file`. Clone with `git lfs` beside the
  XML if you want a meshed model — the bridge reads only body frames, so it
  gains nothing from the meshes.
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

## See also

- [ProtoMotions](protomotions.md) - the tracker that plays this cache, and the
  install line both halves need.
- [Kimodo](kimodo.md) - the kinematic generator whose `qpos` output this bridges.
