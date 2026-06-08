"""Top-level ``strands-robots`` console entrypoint.

Subcommands
-----------
``dashboard``   Launch the real-time web dashboard for the robot mesh.

The CLI is intentionally thin: each subcommand delegates to its owning module
so that ``python -m strands_robots.dashboard`` and ``strands-robots dashboard``
share one implementation.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strands-robots",
        description="AI-powered robot control, simulation, and the robot mesh dashboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dash = sub.add_parser("dashboard", help="Launch the real-time web dashboard.")
    dash.add_argument("--port", "-p", type=int, default=7860, help="TCP port (default: 7860).")
    dash.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 loopback-only; use 0.0.0.0 to expose).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        from strands_robots.dashboard.server import start_dashboard

        start_dashboard(host=args.host, port=args.port)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
