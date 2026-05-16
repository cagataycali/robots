#!/usr/bin/env python3
"""Hardware validation for SO-101 + camera mesh join.

Connects a physical SO-101 arm with a camera, joins the zenoh mesh, and
verifies a *separate* peer can:

1. discover the hardware robot via presence
2. see the advertised camera names
3. (optional) receive frames if ``STRANDS_MESH_CAMERA_HZ`` > 0

Run with two terminals:

    # Terminal A — the hardware peer
    STRANDS_MESH_CAMERA_HZ=2 python scripts/validate_hardware_mesh.py hw \\
        --port /dev/ttyACM0 --camera /dev/video0

    # Terminal B — observer peer
    python scripts/validate_hardware_mesh.py observe

The observer prints a JSON report and exits ``0`` on success, ``1`` on
failure (no peer found within timeout, or no cameras advertised).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

logger = logging.getLogger(__name__)


def cmd_hw(args: argparse.Namespace) -> int:
    """Connect SO-101 hardware + camera, join mesh, hold open."""
    from strands_robots import Robot

    cameras = None
    if args.camera:
        cameras = {
            "wrist": {
                "type": "opencv",
                "index_or_path": args.camera,
                "fps": args.camera_fps,
                "width": 640,
                "height": 480,
            }
        }

    print(f"[hw] connecting SO-101 on {args.port} (camera={args.camera})...")
    hw = Robot(
        "so101",
        mode="real",
        port=args.port,
        cameras=cameras,
    )

    if hw.mesh is None:
        print("[hw] ❌ mesh did not initialize (eclipse-zenoh missing?)")
        return 1

    print(f"[hw] ✅ on mesh as {hw.peer_id}")
    print(f"[hw] heartbeat advertising cameras={list(cameras.keys()) if cameras else []}")
    print("[hw] holding mesh open — Ctrl-C to exit")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[hw] stopping...")
    finally:
        try:
            hw.mesh.stop()
        except Exception:  # noqa: BLE001
            pass

    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    """Observer peer — discover the hardware robot and validate."""
    from strands_robots.mesh import init_mesh
    from strands_robots.mesh_session import get_peers

    class _Probe:
        tool_name_str = "mesh-validator"

    mesh = init_mesh(_Probe(), peer_id="validator", peer_type="agent")
    if mesh is None:
        print("[observe] ❌ mesh did not initialize")
        return 1

    print(f"[observe] joined as {mesh.peer_id}; waiting up to {args.timeout}s for hardware peer...")

    deadline = time.time() + args.timeout
    target = None
    while time.time() < deadline:
        for peer in get_peers():
            if peer.get("connected") and peer.get("hw"):
                target = peer
                break
        if target is not None:
            break
        time.sleep(0.5)

    if target is None:
        print(f"[observe] ❌ no connected hardware peer found within {args.timeout}s")
        mesh.stop()
        return 1

    cameras = target.get("cameras") or []
    report = {
        "peer_id": target.get("peer_id"),
        "type": target.get("robot_type"),
        "hw": target.get("hw"),
        "connected": target.get("connected"),
        "cameras": cameras,
        "instruction": target.get("instruction"),
    }
    print("[observe] discovered hardware peer:")
    print(json.dumps(report, indent=2))

    ok = bool(cameras) if args.require_cameras else True

    if args.frames and cameras:
        # Subscribe to a single camera topic for `args.frames` seconds.
        import threading

        recv = {"count": 0, "first_shape": None}
        ev = threading.Event()
        cam = cameras[0]
        topic = f"strands/{target['peer_id']}/camera/{cam}"

        def on_frame(_topic: str, data: dict) -> None:
            recv["count"] += 1
            if recv["first_shape"] is None:
                recv["first_shape"] = data.get("shape")
            if recv["count"] >= 3:
                ev.set()

        mesh.subscribe(topic, on_frame, name=f"validate-{cam}")
        print(f"[observe] subscribed to {topic} for {args.frames}s...")
        ev.wait(timeout=args.frames)
        mesh.unsubscribe(f"validate-{cam}")
        print(f"[observe] frames received: {recv['count']}, first shape: {recv['first_shape']}")
        ok = ok and recv["count"] > 0

    mesh.stop()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    hw = sub.add_parser("hw", help="run the hardware peer (SO-101 + camera)")
    hw.add_argument("--port", default="/dev/ttyACM0", help="servo bus device path")
    hw.add_argument("--camera", default=None, help="camera path/index, e.g. /dev/video0")
    hw.add_argument("--camera-fps", type=int, default=30)
    hw.set_defaults(func=cmd_hw)

    obs = sub.add_parser("observe", help="run the observer peer")
    obs.add_argument("--timeout", type=float, default=15.0)
    obs.add_argument("--require-cameras", action="store_true")
    obs.add_argument(
        "--frames",
        type=float,
        default=0.0,
        help="seconds to wait for camera frames on the first advertised camera",
    )
    obs.set_defaults(func=cmd_observe)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
