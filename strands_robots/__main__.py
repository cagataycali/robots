"""Entry point for ``python -m strands_robots <command>``."""

from __future__ import annotations

import os
import sys

_COMMANDS = ("doctor", "verify-dataset", "dashboard", "iot")


def _invocation_name() -> str:
    """Name this process was invoked as, for a usage line that matches what the user typed.

    The console script ``strands-robots`` and ``python -m strands_robots`` reach the same
    dispatcher, so hard-coding one name recommends the wrong invocation to a user running
    the other. Fall back to ``strands-robots`` (the documented entry) when ``argv[0]`` has
    been emptied (``python -c``, a frozen entry point) rather than guessing ``python -m``.
    """
    base = os.path.basename(sys.argv[0] or "")
    if base in ("", "__main__.py", "python", "python3") or base.startswith("python"):
        return "python -m strands_robots"
    return base


def main() -> None:
    """Dispatch ``python -m strands_robots <command>`` to its subcommand.

    Routes the first argv token to the ``doctor``, ``verify-dataset``,
    ``dashboard`` or ``iot`` entry point (stripping it so the subcommand parses clean args) and exits non-zero
    on a missing or unknown command.
    """
    if len(sys.argv) < 2:
        # Name the invocation the user actually typed; the console-script and
        # the ``python -m`` entry reach the same dispatcher, so hard-coding one
        # name here recommended the wrong one to half the users (harness#475
        # fixed the sibling ``-h/--help/-V/--version`` branches below; the
        # no-args branch was missed).
        print(f"Usage: {_invocation_name()} <command>")
        print(f"Commands: {', '.join(_COMMANDS)}")
        sys.exit(1)

    cmd = sys.argv[1]
    # The two flags every console script is tried with first. Answering them
    # with "Unknown command" and exit 1 makes a fresh install look broken.
    if cmd in ("-h", "--help"):
        print("Usage: strands-robots <command> [options]")
        print(f"Commands: {', '.join(_COMMANDS)}")
        return
    if cmd in ("-V", "--version"):
        from importlib.metadata import version

        print(f"strands-robots {version('strands-robots')}")
        return
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

        sys.exit(dashboard_main())
    elif cmd == "iot":
        from strands_robots.mesh.iot.cli import main as iot_main

        sys.exit(iot_main())
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
