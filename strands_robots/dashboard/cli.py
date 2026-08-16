"""``strands-robots dashboard`` CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys


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

    mesh = parser.add_argument_group(
        "remote mesh",
        "The dashboard is a mesh peer, not a hub - it can join a mesh running "
        "anywhere. These are persisted to the settings file and editable live "
        "from the UI (Settings -> Mesh).",
    )
    mesh.add_argument(
        "--zenoh-connect", default=None, metavar="EP[,EP...]",
        help="Endpoints to dial, e.g. tls/robot.lan:7447 (sets ZENOH_CONNECT)",
    )
    mesh.add_argument(
        "--zenoh-listen", default=None, metavar="EP[,EP...]",
        help="Endpoints to listen on, e.g. tls/0.0.0.0:7447 (sets ZENOH_LISTEN)",
    )
    mesh.add_argument("--mesh-port", type=int, default=None, help="Mesh port (default 7447)")
    mesh.add_argument(
        "--mesh-backend", default=None, choices=("zenoh", "iot", "bridge"),
        help="Transport: zenoh (LAN/VPN), iot (AWS IoT Core over the internet), bridge",
    )
    mesh.add_argument("--camera-hz", type=float, default=None, help="Camera publish rate cap")

    sec = parser.add_argument_group("security")
    sec.add_argument(
        "--auth-token", default=None, metavar="TOKEN",
        help="Require this bearer token on every /api and /ws request. Without "
             "it the dashboard is open to anyone who can reach the port - and "
             "it moves real motors.",
    )
    sec.add_argument(
        "--cors-origin", action="append", default=None, metavar="ORIGIN",
        help="Allowed CORS origin (repeatable). Default '*'.",
    )
    args = parser.parse_args()

    if args.local_dev:
        os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    from strands_robots.dashboard import settings
    from strands_robots.dashboard.mesh_bridge import MeshBridge
    from strands_robots.dashboard.server import create_app

    # CLI flags are persisted, so they behave the same as the same fields set in
    # the UI - one resolution order (settings -> env -> default), not two.
    patch: dict[str, dict[str, object]] = {"mesh": {}, "security": {}}
    if args.zenoh_connect is not None:
        patch["mesh"]["connect"] = args.zenoh_connect
    if args.zenoh_listen is not None:
        patch["mesh"]["listen"] = args.zenoh_listen
    if args.mesh_port is not None:
        patch["mesh"]["port"] = args.mesh_port
    if args.mesh_backend is not None:
        patch["mesh"]["backend"] = args.mesh_backend
    if args.camera_hz is not None:
        patch["mesh"]["camera_hz"] = args.camera_hz
    if args.auth_token is not None:
        patch["security"]["auth_token"] = args.auth_token
    if args.cors_origin is not None:
        patch["security"]["cors_origins"] = args.cors_origin
    changed = settings.update({k: v for k, v in patch.items() if v})

    resolved = settings.load(refresh=True)
    mesh_cfg = resolved["mesh"]
    local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")

    print(f"🤖 strands-robots dashboard → http://{args.host}:{args.port}")
    print(f"   mesh: backend={mesh_cfg.get('backend') or 'zenoh'} "
          f"port={mesh_cfg.get('port') or 7447} "
          f"connect={mesh_cfg.get('connect') or ['<multicast/local>']} "
          f"listen={mesh_cfg.get('listen') or ['<default>']}")
    if local_dev:
        print("   ⚠️  STRANDS_MESH_LOCAL_DEV=1 - WIRE SECURITY DISABLED (no TLS, no auth)")
    if resolved["security"].get("auth_token"):
        print("   auth: bearer token required on /api and /ws")
    else:
        print("   ⚠️  no auth token - anyone who can reach this port can move motors "
              "(--auth-token to require one)")
    if changed:
        print(f"   saved to {settings.SETTINGS_FILE}: {', '.join(changed)}")
    # Piped stdout is block-buffered, so without this the security warnings land
    # *after* uvicorn's log lines - or only at exit - in a redirected logfile.
    sys.stdout.flush()

    app = create_app(MeshBridge(peer_id=args.peer_id))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
