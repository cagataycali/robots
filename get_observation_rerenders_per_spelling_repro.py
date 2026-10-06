"""
Minimal repro: `_get_sim_observation` rerenders a camera for every spelling
=========================================================================

Target:  strands-labs/robots v0.5.3 (verified at upstream HEAD 1ba597d19)
File:    strands_robots/simulation/mujoco/rendering.py:794-842
Class:   Silent-wrong + Wasted work (asymmetric guard)
Rotation target: lekiwi_sim_quickstart (ships cameras under short alias)

TL;DR
-----
Every multi-robot-namespaced camera is rendered TWICE per
`get_observation` call: once under the model-named key (`lekiwi/front`),
once under the short alias (`front`). Both keys name the same compiled
camera id (`_camera_id` resolves them identically), but the current loop
issues a second `update_scene` + `render` pair for the second spelling.

Numbers (`model.ncam` vs camera-render passes vs resulting obs image keys):

    lekiwi    3 model cams -> 5 render passes -> 5 obs image keys
    stretch   3 model cams -> 5 render passes -> 5 obs image keys
    stretch3  6 model cams -> 11 render passes -> 11 obs image keys
    so100     1 model cam  -> 1 render pass   -> 1 obs image key   (control)
    so101     1 model cam  -> 1 render pass   -> 1 obs image key   (control)

Why this is silent-wrong (not just "a cost footgun")
----------------------------------------------------
The two renders produce two separate `numpy` buffers, written by two
calls to `renderer.update_scene` + `renderer.render` on the same tick.
They are NOT guaranteed to be bytewise equal:

    stretch3/d405_rgb        hash 7eb5eb67b376   DIFFERENT
    d405_rgb                 hash 60f1c21d79c8

    stretch3/d405_depth      hash 7458994fa11b   DIFFERENT
    d405_depth               hash 3fd3ef9b8029

    stretch3/d435i_camera_rgb        hash 8ebb6c32da4c   same (cache coincidence)
    d435i_camera_rgb                 hash 8ebb6c32da4c

Which of the two keys a user reads is undefined by the project doc, so a
caller who writes `camera_keys=["d405_rgb"]` and a caller who writes
`camera_keys=["stretch3/d405_rgb"]` record DIFFERENT images while both
pass every schema guard. The dataset recorder cannot catch it: both keys
are valid, both carry the right shape, both resolve to a `cam_id` the
scene really has. The two-camera-two-views pin in
`tests/simulation/mujoco/test_observation_camera_keys_carry_their_own_view.py`
assumes one `render.render()` call per cam_id spelling, not that the view
is deterministic across the two spellings of one cam_id.

The schema is also pinned to publish BOTH spellings
(same test file: `test_short_key_carries_the_robots_own_camera_view` on
the short, `test_model_named_keys_are_unaffected` on the namespaced), so
"drop one" would break the pin -- the fix below renders ONCE per cam_id
and publishes the one buffer under both keys. Bytewise-equal keys for
the same cam_id, no extra render pass.

Expected (after fix)
--------------------

    stretch3/d405_rgb and d405_rgb       SAME numpy buffer (zero-copy)
    stretch3/d405_depth and d405_depth   SAME numpy buffer
    both schema pins still satisfied     SAME behaviour from policy / recorder
    render passes per get_observation()  halves on every namespaced robot

Fix sketch (strands_robots/simulation/mujoco/rendering.py:794-842)
------------------------------------------------------------------
Cache the render result by `cam_id` inside the loop; publish the cached
view under every subsequent key that resolves to the same `cam_id`:

    rendered_views: dict[int, np.ndarray] = {}
    for cname in cameras_to_render:
        ...
        cam_id = self._camera_id(cname)
        if cam_id < 0: ...
        ...
        if cam_id in rendered_views:
            obs[cname] = rendered_views[cam_id]
            continue
        ...
        frame = renderer.render().copy()
        obs[cname] = frame
        rendered_views[cam_id] = frame

~10 LOC. The pinned tests (5 in
`test_observation_camera_keys_carry_their_own_view.py`) still pass.

Run
---
    python get_observation_doubles_namespaced_cameras_repro.py
"""
from __future__ import annotations

import hashlib
import sys

import mujoco
import numpy as np

from strands_robots import Robot


def _image_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def main() -> int:
    print("=" * 72)
    print("Repro: _get_sim_observation rerenders a camera per spelling")
    print("=" * 72)

    rc = 0

    for robot_name in ("lekiwi", "stretch", "stretch3", "so100", "so101"):
        print(f"\n--- {robot_name} ---")
        r = Robot(robot_name, mesh=False)
        model = r.mj_model

        model_cams = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        print(f"MuJoCo model.ncam = {model.ncam}")
        print(f"MuJoCo camera names: {model_cams}")

        obs = r.get_observation()
        img_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3]
        print(f"get_observation() image keys ({len(img_keys)}): {img_keys}")

        # Every (prefix/short, short) pair must now share the same numpy buffer
        # (zero-copy) -- that is the shape of the fix. Pre-fix: separate buffers
        # whose BYTES may differ.
        any_pair_checked = False
        for pk in list(img_keys):
            if "/" not in pk:
                continue
            short = pk.rsplit("/", 1)[-1]
            if short not in img_keys:
                continue
            any_pair_checked = True
            cid_prefix = r._camera_id(pk)
            cid_short = r._camera_id(short)
            same_bytes = np.array_equal(obs[pk], obs[short])
            same_buffer = obs[pk] is obs[short]
            print(
                f"  {pk!r} vs {short!r}: "
                f"cam_id(prefix)={cid_prefix}  cam_id(short)={cid_short}  "
                f"same_bytes={same_bytes}  same_buffer={same_buffer}  "
                f"hash_prefix={_image_hash(obs[pk])}  hash_short={_image_hash(obs[short])}"
            )
            if cid_prefix == cid_short and cid_prefix >= 0:
                if not same_bytes:
                    rc = 1  # silent-wrong: two keys, one cam_id, different pixels
                elif not same_buffer:
                    # bytes equal but two buffers: double render that happened
                    # to land on the same pixels. Still 2x the render cost; the
                    # fix publishes a single buffer under both keys.
                    rc = 1

        if not any_pair_checked:
            print("  (no alias pairs to check -- single-camera robot, control)")

    print()
    print("=" * 72)
    if rc:
        print("FAILED: _get_sim_observation re-renders cameras per spelling.")
        print(
            "        Fix: cache the first render per cam_id and publish the"
        )
        print(
            "        cached buffer under every subsequent key that resolves"
        )
        print("        to the same cam_id. Keeps both pinned schema keys.")
    else:
        print(
            "PASSED: each cam_id is rendered once per get_observation() call "
            "and the one buffer is published under every key that resolves to it."
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
