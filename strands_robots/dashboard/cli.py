"""``strands-robots dashboard`` CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="strands-robots dashboard",
        description="Fleet cockpit for the strands-robots mesh (PWA + API).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--peer-id", default=None, help="Mesh peer id for the dashboard (auto if omitted)")
    parser.add_argument("--local-dev", action="store_true", help="Set STRANDS_MESH_LOCAL_DEV=1 (no TLS, single machine)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.local_dev:
        os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    from strands_robots.dashboard.mesh_bridge import MeshBridge
    from strands_robots.dashboard.server import create_app

    app = create_app(MeshBridge(peer_id=args.peer_id))
    print(f"🤖 strands-robots dashboard → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
