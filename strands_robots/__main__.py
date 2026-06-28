"""Entry point for ``python -m strands_robots <command>``."""

from __future__ import annotations

import sys

_COMMANDS = ("doctor", "verify-dataset")


def main() -> None:
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
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
