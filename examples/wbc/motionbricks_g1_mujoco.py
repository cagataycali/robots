#!/usr/bin/env python3
"""MotionBricks generative motion on the Unitree G1 in MuJoCo (headless).

Drives the :class:`~strands_robots.policies.motionbricks.MotionBricksPolicy`
through a style sequence (walk -> stealth_walk -> walk_boxing) and records a
kinematic rollout MP4. MotionBricks is a *kinematic* motion generator: each tick
it synthesises a full-body ``qpos`` (root + joints). The faithful way to
visualise it - matching the upstream ``interactive_demo_g1.py`` - is to set that
``qpos`` on the model and run forward kinematics (no physics step). The policy's
action dict carries the 29 joint targets (the signal a tracking controller such
as :class:`~strands_robots.policies.wbc.WBCPolicy` would consume under physics);
the example reads the accompanying root pose from ``policy.last_qpos`` so the
character translates through the world as generated.

Usage (headless, on a machine with the upstream checkpoints fetched via
git-LFS)::

    MUJOCO_GL=egl python examples/wbc/motionbricks_g1_mujoco.py \
        --result-dir /path/to/GR00T-WholeBodyControl/motionbricks/out \
        --device cuda --out /tmp/motionbricks_g1.mp4

``--result-dir`` defaults to the ``MOTIONBRICKS_CKPT`` environment variable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default=os.environ.get("MOTIONBRICKS_CKPT", ""),
        help="Path to the upstream MotionBricks 'out/' checkpoint tree (default: $MOTIONBRICKS_CKPT).",
    )
    parser.add_argument("--scene-xml", default="", help="G1 scene XML (default: derived from --result-dir).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device for the generator.")
    parser.add_argument("--out", default="/tmp/motionbricks_g1.mp4", help="Output MP4 path.")
    parser.add_argument("--steps-per-style", type=int, default=80, help="Control steps per style segment.")
    parser.add_argument("--fps", type=int, default=30, help="Render / motion FPS.")
    parser.add_argument(
        "--styles",
        default="walk,stealth_walk,walk_boxing",
        help="Comma-separated clip styles to cycle through.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from strands_robots.policies.motionbricks import MotionBricksConfig, MotionBricksPolicy

    if not args.result_dir:
        raise SystemExit(
            "Pass --result-dir (or set MOTIONBRICKS_CKPT) to the upstream MotionBricks 'out/' "
            "checkpoint tree. Fetch it with:\n"
            '  git lfs pull --include="motionbricks/out/**" --exclude=""'
        )
    result_dir = Path(args.result_dir).expanduser().resolve()
    scene_xml = args.scene_xml or str(result_dir.parent / "assets" / "skeletons" / "g1" / "scene_29dof.xml")

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    print(f"Building MotionBricks generator from {result_dir} on {args.device} ...")
    config = MotionBricksConfig(result_dir=str(result_dir), device=args.device, fps=args.fps)
    policy = MotionBricksPolicy(config=config, style=styles[0])

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 4.0
    camera.elevation = -15.0

    policy.reset()
    frames: list[np.ndarray] = []
    print(f"Rolling out styles {styles} ({args.steps_per_style} steps each) ...")
    for style in styles:
        for _ in range(args.steps_per_style):
            policy.get_actions_sync({}, "", style=style)
            qpos = policy.last_qpos
            assert qpos is not None
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            # Track the character with the free camera.
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

    renderer.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=args.fps)
    print(f"Wrote {len(frames)} frames to {out} ({args.fps} fps, styles={styles}).")


if __name__ == "__main__":
    main()
