"""Entry point for ``python -m strands_robots <command>``."""

from __future__ import annotations

import sys

_COMMANDS = ("doctor", "verify-dataset", "dashboard", "dev")


def main() -> None:
    """Dispatch ``python -m strands_robots <command>`` to its subcommand.

    Routes the first argv token to the ``doctor`` or ``verify-dataset`` entry
    point (stripping it so the subcommand parses clean args) and exits non-zero
    on a missing or unknown command.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m strands_robots <command>")
        print(f"Commands: {', '.join(_COMMANDS)}")
        sys.exit(1)

    cmd = sys.argv[1]
    # Remove the command from argv so sub-parsers see clean args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "doctor":
        from strands_robots.doctor import main as doctor_main

        doctor_main()
    elif cmd == "verify-dataset":
        from strands_robots.verify_dataset import main as verify_main

        sys.exit(verify_main())
    elif cmd == "dashboard":
        from strands_robots.dashboard.cli import main as dashboard_main

        dashboard_main()
    elif cmd == "dev":
        from strands_robots.dashboard.dev import main as dev_main

        sys.exit(dev_main())
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
