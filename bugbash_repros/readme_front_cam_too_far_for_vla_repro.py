#!/usr/bin/env python3
"""
Reproducer: README quickstart's `front` camera is placed too far away for a
VLA / vision-language-action model to actually see the 5 cm red_cube.

Context
-------
After harness #725 (closed, "front camera renders 0 red pixels — gripper
occludes the cube"), the README was updated to move the camera from
(0.0, -0.5, 0.3) to (0.3, -0.7, 0.45). This clears the occlusion, but it
also moves the camera from 0.3 m away to **0.72 m away** from the cube.

The paragraph two blocks above the quickstart (README.md:41-47) advertises
VLA policies:

    "Learned policies from the Hugging Face Hub, from vision-language-action
    models to world foundation models and whole-body controllers, run through
    the same `run_policy` call in the simulator and on the physical robot."

The quickstart's own code sets the scene up for exactly this - but the
`front` camera at 0.72 m renders the 5 cm cube at **12 x 6 pixels (0.018 %
of frame)**, below the useful threshold for any VLM (16 x 16 is the
rule-of-thumb lower bound; LeRobot's own smolVLA ingests 384-wide crops).

The eight canonical examples (`examples/01_sim_hello_world.py`,
`examples/02_policy_abstraction.py`, `examples/03_record_dataset.py`,
`examples/07_post_tune_any_policy.py`, ...) all use a camera at
`position=[0.5, 0.0, 0.4]`, `target=[0.2, 0, 0.05]` for a cube at
`[0.2, 0.0, 0.05]` - 0.52 m distance, which renders a 2.5 cm cube
large enough to be a usable VLA input. The README deviates from this
convention without a documented reason.

Run
---
    MUJOCO_GL=egl python bugbash_repros/readme_front_cam_too_far_for_vla_repro.py

Expected output
---------------
    README front camera (0.3, -0.7, 0.45): 54 red pixels, bbox 12x6 (0.018%)
      -> TOO SMALL for VLA (bbox w/h both < 16 px)
    examples/01 convention (0.5,  0.0, 0.4): >= 400 red pixels, bbox >= 30x30 (>0.13%)
      -> VISIBLE
    FAIL: README quickstart scene renders the advertised VLA target below
    useful resolution; the canonical examples do not.
"""
from __future__ import annotations

import sys

import numpy as np

from strands_robots import Robot


def _red_bbox(img: np.ndarray) -> tuple[int, int, int, float]:
    """Return (n_red_pixels, bbox_w, bbox_h, area_fraction)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (r > 200) & (g < 80) & (b < 80)
    n = int(mask.sum())
    if n == 0:
        return 0, 0, 0, 0.0
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min())
    h = int(ys.max() - ys.min())
    frac = n / float(img.shape[0] * img.shape[1])
    return n, w, h, frac


def _render_scene(cube_pos, cube_size, cam_pos, cam_target):
    """Render a Robot('so100') scene with one object + one camera, return the
    front camera image."""
    r = Robot("so100", mesh=False)
    r.add_object(
        name="red_cube",
        shape="box",
        size=cube_size,
        position=cube_pos,
        color=[1.0, 0.0, 0.0],
    )
    r.add_camera(name="front", position=cam_pos, target=cam_target)
    obs = r.get_observation()
    return obs["front"]


def main() -> int:
    # README.md:53-56 quickstart - verbatim.
    readme_img = _render_scene(
        cube_pos=[0.0, -0.2, 0.025],
        cube_size=[0.05, 0.05, 0.05],
        cam_pos=[0.3, -0.7, 0.45],
        cam_target=[0.0, -0.2, 0.03],
    )
    n_r, w_r, h_r, f_r = _red_bbox(readme_img)
    print(
        f"README front camera (0.3, -0.7, 0.45): {n_r} red pixels, "
        f"bbox {w_r}x{h_r} ({f_r * 100:.3f}%)"
    )
    readme_too_small = w_r < 16 or h_r < 16
    print(f"  -> {'TOO SMALL for VLA (bbox w/h both < 16 px)' if readme_too_small else 'VISIBLE'}")

    # examples/01_sim_hello_world.py:21-29 scene - same `so100`, same camera
    # name, different (canonical) placement.
    ex01_img = _render_scene(
        cube_pos=[0.2, 0.0, 0.05],
        cube_size=[0.025, 0.025, 0.025],
        cam_pos=[0.5, 0.0, 0.4],
        cam_target=[0.2, 0.0, 0.05],
    )
    n_e, w_e, h_e, f_e = _red_bbox(ex01_img)
    print(
        f"examples/01 convention (0.5,  0.0, 0.4): {n_e} red pixels, "
        f"bbox {w_e}x{h_e} ({f_e * 100:.3f}%)"
    )
    ex01_visible = w_e >= 16 and h_e >= 16
    print(f"  -> {'VISIBLE' if ex01_visible else 'TOO SMALL for VLA'}")

    if readme_too_small and ex01_visible:
        print(
            "FAIL: README quickstart scene renders the advertised VLA target "
            "below useful resolution; the canonical examples do not."
        )
        return 1
    print("PASS: README and examples agree on usable VLA cube visibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
