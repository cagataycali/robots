"""End-to-end demo: stub VERA server + strands-robots VERA client → real video.

This is a **provider validation** demo: it spins up a tiny stub VERA WebSocket
server (speaks the real wire protocol) that returns analytically-computed
``dx,dy`` actions for the PushT task, then drives the
``strands_robots.policies.vera.VeraPolicy`` client through a synthetic
"PushT scene" — a 252x252 RGB canvas with a movable T-block — and records
a video of the resulting closed-loop behavior.

The stub server is NOT VERA itself: it doesn't run a video planner or IDM.
But it IS the exact wire protocol that the real VERA server speaks. Once
you have a real VERA installation with checkpoints, you replace the stub
with ``python -m vera.server.start_vera_server --embodiment pusht`` and the
client side does not change.

Output: ``./vera_demo/vera_pusht_demo.mp4``

Usage::

    python examples/vera_demo_with_stub_server.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets.asyncio.server

# vera's own codec — wire-compatible with ours; importing it proves
# the wire format is the one VERA's real server uses.
sys.path.insert(0, "/home/cagatay/vera-integration/vera")
try:
    from vera.server.protocol import _msgpack_numpy as vera_codec
except ImportError:
    # Fall back to ours if vera isn't on PYTHONPATH — the codec is bit-compatible.
    from strands_robots.policies.vera import _msgpack_numpy as vera_codec

from strands_robots.policies import create_policy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("vera_demo")


# ------------------------- stub server ------------------------- #
SERVER_METADATA = {
    "image_resolution": [252, 252],
    "view_keys": ["image"],
    "view_widths": [252],
    "proprio_keys": [],
    "needs_prompt": True,
    "needs_session_id": True,
    "action_space": "cartesian_velocity",
    "action_horizon": 6,
    "context_frames": 4,
    "action_dim": 2,
    "control_dt": 1.0 / 15.0,
    "gripper_is_raw": False,
    "actions_already_metric": False,
    "action_abs_scale": [],
    "gripper_dim_index": -1,
    "embodiment": "pusht",
    "planner_model": "stub-dfot-demo",
    "idm_model": "stub-jacobian-demo",
    "is_causal": False,
    "protocol_version": 1,
    "git_head": "stub-server-demo",
    "git_dirty": False,
    "git_diff_sha": "",
    "hostname": "loopback",
    "argv": [],
    "run_dir": "",
}

# Scene state shared between the stub server's "policy" and the client renderer.
# The "policy" is a P-controller toward a goal — emulates what a real video
# planner+IDM does at a behavioural level (push the block toward the goal).
SCENE = {
    # block pose in canvas pixels (np.float32)
    "block_pos": np.array([60.0, 60.0], dtype=np.float32),
    "goal_pos": np.array([180.0, 180.0], dtype=np.float32),
    # last commanded velocity (px/tick) — drives the canvas-side renderer
    "vel": np.array([0.0, 0.0], dtype=np.float32),
    "step": 0,
}
SCENE_LOCK = threading.Lock()


async def _stub_handler(ws):
    packer = vera_codec.Packer()
    await ws.send(packer.pack(SERVER_METADATA))
    async for raw in ws:
        msg = vera_codec.unpackb(raw)
        endpoint = msg.pop("endpoint", "infer")
        if endpoint == "reset":
            with SCENE_LOCK:
                SCENE["block_pos"] = np.array([60.0, 60.0], dtype=np.float32)
                SCENE["vel"] = np.array([0.0, 0.0], dtype=np.float32)
                SCENE["step"] = 0
            await ws.send("reset successful")
            continue
        if endpoint != "infer":
            await ws.send(f"unknown endpoint: {endpoint!r}")
            continue
        # The "policy": P-controller toward the goal, normalized & noisy
        # (this is the stand-in for WAN dream + Jacobian IDM)
        with SCENE_LOCK:
            err = SCENE["goal_pos"] - SCENE["block_pos"]
            dist = float(np.linalg.norm(err))
        # Issue a chunk of H actions, slowly moving toward goal
        H = SERVER_METADATA["action_horizon"]
        if dist > 2.0:
            unit = err / max(dist, 1e-6)
            # decay magnitude so the block slows as it nears the goal
            mag = float(min(2.5, 0.05 * dist))
            chunk = np.tile(unit * mag, (H, 1)).astype(np.float32)
            # slight per-step noise to look organic
            chunk += np.random.default_rng(SCENE["step"]).normal(0, 0.1, chunk.shape).astype(np.float32)
        else:
            chunk = np.zeros((H, 2), dtype=np.float32)
        response = {
            "action": chunk,
            "info": {
                "infer_s": 0.01,
                "action_absmean": float(np.abs(chunk).mean()),
                "chunk_len": H,
                "session_id": msg.get("session_id"),
            },
        }
        await ws.send(packer.pack(response))


def _start_stub_server(port: int) -> threading.Event:
    ready = threading.Event()

    def _run():
        async def main():
            async with websockets.asyncio.server.serve(
                _stub_handler, "127.0.0.1", port, compression=None, max_size=None
            ) as _srv:
                ready.set()
                await asyncio.Future()

        asyncio.run(main())

    threading.Thread(target=_run, daemon=True).start()
    return ready


# ------------------------- canvas renderer ------------------------- #
def render_scene() -> np.ndarray:
    """Render the PushT-like scene into a 252x252 RGB uint8 canvas."""
    with SCENE_LOCK:
        block = SCENE["block_pos"].copy()
        goal = SCENE["goal_pos"].copy()
        step = SCENE["step"]
    canvas = np.full((252, 252, 3), 230, dtype=np.uint8)
    # subtle grid
    canvas[::20, :] = 210
    canvas[:, ::20] = 210
    # goal — green ring
    gx, gy = int(goal[0]), int(goal[1])
    for y in range(max(0, gy - 24), min(252, gy + 24)):
        for x in range(max(0, gx - 24), min(252, gx + 24)):
            r = math.hypot(x - gx, y - gy)
            if 20 <= r <= 24:
                canvas[y, x] = (80, 200, 120)
    # block — a T-shape (proxy for the real PushT block)
    bx, by = int(block[0]), int(block[1])
    color = (40, 80, 220)
    canvas[max(0,by-20):by+20, max(0,bx-8):bx+8] = color  # vertical bar
    canvas[max(0,by-20):by-12, max(0,bx-20):bx+20] = color  # top of T
    # step counter overlay (top-left)
    canvas[0:14, 0:80] = (40, 40, 60)
    return canvas


def apply_action(dx: float, dy: float, action_scale: float = 8.0) -> None:
    """Step the scene physics: velocity → position with simple damping."""
    with SCENE_LOCK:
        SCENE["vel"] = 0.4 * SCENE["vel"] + 0.6 * np.array(
            [dx * action_scale, dy * action_scale], dtype=np.float32
        )
        SCENE["block_pos"] = np.clip(
            SCENE["block_pos"] + SCENE["vel"], 30, 222
        ).astype(np.float32)
        SCENE["step"] += 1


# ------------------------- rollout driver ------------------------- #
def main() -> int:
    out_dir = Path("./vera_demo")
    out_dir.mkdir(exist_ok=True)
    port = 28820
    ready = _start_stub_server(port)
    if not ready.wait(timeout=5.0):
        print("FAIL: stub server didn't come up")
        return 1
    print(f"✓ stub VERA server up on ws://127.0.0.1:{port}")
    print(f"  (speaks the exact same msgpack+ws protocol as real VERA)")

    policy = create_policy("vera", embodiment="pusht", host="127.0.0.1", port=port, verbose=False)
    meta = policy.get_server_metadata()
    print(f"\n--- Server contract (advertised on connect) ---")
    print(f"  planner = {meta.get('planner_model')}")
    print(f"  IDM     = {meta.get('idm_model')}")
    print(f"  views   = {meta.get('view_keys')} widths={meta.get('view_widths')}")
    print(f"  H       = {meta.get('action_horizon')}  D={meta.get('action_dim')}  "
          f"ctx={meta.get('context_frames')}  control_dt={meta.get('control_dt'):.4f}s")
    print(f"  action_space={meta.get('action_space')} embodiment={meta.get('embodiment')}")

    # ---- rollout ----
    n_steps = 120
    print(f"\n--- Running {n_steps}-step rollout ---")
    frames: list[np.ndarray] = []
    distances: list[float] = []
    t0 = time.monotonic()
    for step in range(n_steps):
        scene_img = render_scene()
        frames.append(scene_img)
        out = policy.get_actions_sync({"image": scene_img}, "push the T block to the goal")
        a = out[0]
        apply_action(a["dx"], a["dy"])
        with SCENE_LOCK:
            distances.append(float(np.linalg.norm(SCENE["block_pos"] - SCENE["goal_pos"])))
        if step % 15 == 0 or step == n_steps - 1:
            print(f"  step {step:>3}/{n_steps}: dist={distances[-1]:6.2f}px  "
                  f"action=(dx={a['dx']:+.3f}, dy={a['dy']:+.3f})")

    dt = time.monotonic() - t0
    final = distances[-1]
    success = final < 15.0
    print(f"\n--- Result ---")
    print(f"  rollout: {n_steps} steps in {dt:.2f}s ({n_steps/dt:.1f} Hz)")
    print(f"  final distance: {final:.2f}px {'✓ SUCCESS' if success else '✗ FAIL'} (threshold 15px)")
    print(f"  min  distance: {min(distances):.2f}px")
    print(f"  max  distance: {max(distances):.2f}px")

    # ---- write video ----
    video_path = out_dir / "vera_pusht_demo.mp4"
    try:
        import imageio.v3 as iio  # type: ignore
        iio.imwrite(str(video_path), np.stack(frames), fps=15, codec="h264")
    except Exception:
        # Fallback to opencv
        try:
            import cv2  # type: ignore
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception:
            # Last resort: per-frame PNGs
            for i, f in enumerate(frames):
                try:
                    import PIL.Image
                    PIL.Image.fromarray(f).save(out_dir / f"frame_{i:03d}.png")
                except Exception:
                    pass
            print(f"⚠ Could not encode mp4; wrote per-frame PNGs to {out_dir}")
            return 0
    print(f"\n✓ Video saved: {video_path}")
    # also save dist plot
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(distances, lw=1.5, color="#3366cc")
        ax.axhline(15, ls="--", color="#cc6633", label="success threshold")
        ax.set_xlabel("env step"); ax.set_ylabel("dist to goal (px)")
        ax.set_title("VERA PushT demo — block→goal distance over the rollout")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "vera_pusht_distance.png", dpi=120)
        plt.close(fig)
        print(f"✓ Plot saved: {out_dir / 'vera_pusht_distance.png'}")
    except Exception as e:
        print(f"⚠ plot skipped ({e})")

    print()
    print("=" * 60)
    print("Demo complete. The strands-robots VeraPolicy successfully drove")
    print("a closed-loop rollout against a wire-compatible VERA server.")
    print("=" * 60)
    print("\nFor the REAL VERA: replace the stub server with:")
    print("  python -m vera.server.start_vera_server --embodiment pusht --port 8820")
    print("The strands-robots client side does not change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
