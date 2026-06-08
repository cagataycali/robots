"""``python -m strands_robots.dashboard`` entrypoint."""

from __future__ import annotations

import argparse

from strands_robots.dashboard.server import start_dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m strands_robots.dashboard",
        description="Real-time web dashboard for the Strands robot mesh.",
    )
    parser.add_argument("--port", "-p", type=int, default=7860, help="TCP port (default: 7860)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 loopback-only; use 0.0.0.0 to expose).",
    )
    args = parser.parse_args(argv)
    start_dashboard(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
