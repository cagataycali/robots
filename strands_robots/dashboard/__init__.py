"""Strands Robots Dashboard - fleet cockpit for the robot mesh."""

__all__ = ["main"]


def main() -> None:
    from strands_robots.dashboard.cli import main as cli_main

    cli_main()
