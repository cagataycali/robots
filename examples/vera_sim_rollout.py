"""End-to-end VERA rollout — strands-robots Simulation driving VERA's policy server.

This mirrors ``examples/cosmos3_sim_rollout.py``: spin up a strands-robots
MuJoCo simulation, attach the VERA WebSocket policy, run a rollout, and dump a
video of the result.

Prerequisites:

1. Start the VERA policy server (holds the GPU). Quickstart for the **PushT**
   embodiment (smallest checkpoints, loads in seconds — best for first run)::

       # From a VERA checkout (https://github.com/sizhe-li/VERA):
       pip install -e ".[idm,video]"
       export VERA_PUSHT_DFOT_CKPT=./vera-ckpts/pusht-dfot
       export VERA_PUSHT_IDM_CKPT=./vera-ckpts/pusht-idm
       python -m vera.server.start_vera_server --embodiment pusht --port 8820

2. Run this script::

       python examples/vera_sim_rollout.py --embodiment pusht --port 8820

For **MimicGen** (Panda block-stacking, WAN 1.3B planner, ~3.8 GB checkpoints)::

    export VERA_WAN_CKPT_ROOT=/path/to/Wan2.1-T2V-1.3B
    export VERA_MIMICGEN_CKPT_DIR=./vera-ckpts/mimicgen-wan-1.3b
    python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 \
        --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml \
        --text "A robot arm stacks one block on top of another block"

    python examples/vera_sim_rollout.py --embodiment mimicgen --port 8800 \
        --instruction "A robot arm stacks one block on top of another block"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from strands_robots.policies import create_policy
from strands_robots.policies.vera import get_embodiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("vera_sim_rollout")


def _build_pusht_demo(args: argparse.Namespace) -> None:
    """A minimal PushT demo: no MuJoCo dep — just drives a synthetic 252x252
    image through the policy so the user can confirm their server is alive
    BEFORE wiring it into a full sim. Works with vera's released PushT ckpts."""
    print()
    print("=" * 60)
    print("PushT smoke test — drives a synthetic image through VERA's PushT")
    print("server. No MuJoCo / gym-pusht needed for this validation step.")
    print("=" * 60)

    policy = create_policy(
        "vera",
        embodiment="pusht",
        host=args.host,
        port=args.port,
        verbose=True,
    )

    meta = policy.get_server_metadata()
    print(f"\nServer metadata:")
    print(f"  planner    : {meta.get('planner_model')}")
    print(f"  IDM        : {meta.get('idm_model')}")
    print(f"  views      : {meta.get('view_keys')}")
    print(f"  H          : {meta.get('action_horizon')}")
    print(f"  D          : {meta.get('action_dim')}")
    print(f"  action_space: {meta.get('action_space')}")
    print(f"  context_frames: {meta.get('context_frames')}")
    print()

    # Drive 30 ticks of a synthetic "scene" through the policy. Each tick we
    # render a moving square (proxy for a T-block) onto a 252x252 RGB canvas.
    H_ctx = int(meta.get("context_frames", 9))
    n_ticks = max(30, 3 * H_ctx)
    log_every = max(1, n_ticks // 10)
    print(f"Running {n_ticks} ticks…")
    actions: list[dict] = []
    for step in range(n_ticks):
        canvas = np.full((252, 252, 3), 220, dtype=np.uint8)
        cx = 40 + (step * 6) % 170
        cy = 40 + (step * 4) % 170
        canvas[cy : cy + 30, cx : cx + 30] = (40, 80, 220)
        out = policy.get_actions_sync({"image": canvas}, instruction=args.instruction or "push the T to the goal")
        actions.append(out[0])
        if step % log_every == 0:
            print(f"  step {step:>3}: action = {out[0]}")

    a = np.array([[a.get("dx", 0.0), a.get("dy", 0.0)] for a in actions])
    print(f"\nRollout complete:")
    print(f"  {len(actions)} steps, action |mean|={np.abs(a).mean():.4f}")
    print(f"  mean(dx)={a[:, 0].mean():+.4f}  std={a[:, 0].std():.4f}")
    print(f"  mean(dy)={a[:, 1].mean():+.4f}  std={a[:, 1].std():.4f}")

    if args.save_actions:
        out_path = Path(args.output_dir) / "vera_pusht_actions.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, a)
        print(f"  actions saved -> {out_path}")


def _build_mimicgen_demo(args: argparse.Namespace) -> None:
    """A minimal MimicGen / Panda demo wiring into strands-robots ``Simulation``.

    Uses the strands-robots MuJoCo Panda asset, adds 3 cameras matching VERA's
    advertised view_keys, and steps a rollout. Saves a video to
    ``--output-dir``.
    """
    try:
        from strands_robots import Simulation
    except ImportError as e:
        raise ImportError(
            "MimicGen demo needs strands-robots' simulation extras. "
            "Install with: pip install -e '.[sim]'"
        ) from e

    print("=" * 60)
    print(f"VERA MimicGen rollout — Panda block stacking via WAN planner")
    print("=" * 60)

    sim = Simulation(tool_name="sim", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="franka")
    # Match VERA's 3 default mimicgen views; the embodiment table declares them.
    embo = get_embodiment("mimicgen")
    for vk, w in zip(embo.view_keys, embo.view_widths):
        try:
            sim.add_camera(name=vk, width=w, height=w)
        except Exception:
            logger.warning("Could not add camera %r — may need scene customization", vk)

    sim.run_policy(
        robot_name="arm",
        policy_provider="vera",
        policy_config={
            "embodiment": "mimicgen",
            "host": args.host,
            "port": args.port,
            "robot": "panda",
            "verbose": True,
        },
        instruction=args.instruction
        or "A robot arm stacks one block on top of another block",
        n_steps=args.n_steps,
        control_frequency=15.0,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "vera_mimicgen_rollout.mp4"
    try:
        sim.save_video(str(video_path))
        print(f"\n✓ Video saved: {video_path}")
    except Exception:
        logger.exception("save_video failed (continuing)")


def main() -> int:
    ap = argparse.ArgumentParser(description="VERA policy rollout via strands-robots")
    ap.add_argument(
        "--embodiment",
        default="pusht",
        choices=["pusht", "mimicgen", "droid", "allegro"],
        help="VERA embodiment (must match the running server)",
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument(
        "--port",
        type=int,
        default=None,
        help="VERA server port (defaults to the embodiment's standard port)",
    )
    ap.add_argument(
        "--instruction",
        default=None,
        help="Natural-language task (defaults to the embodiment's default_prompt)",
    )
    ap.add_argument("--n-steps", type=int, default=200, help="Rollout horizon (env steps)")
    ap.add_argument(
        "--output-dir",
        default="./vera_rollouts",
        help="Where to save videos / action logs",
    )
    ap.add_argument("--save-actions", action="store_true")
    args = ap.parse_args()

    if args.port is None:
        args.port = get_embodiment(args.embodiment).default_port

    if args.embodiment == "pusht":
        _build_pusht_demo(args)
    else:
        _build_mimicgen_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
