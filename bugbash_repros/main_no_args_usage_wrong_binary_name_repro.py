"""Minimal repro: ``strands-robots`` with no args recommends ``python -m strands_robots``.

The docs promise one console script (``docs/reference/cli.md:7``):
    strands-robots --help       # usage and the command list

And ``strands-robots --help`` already prints the correct name
(``strands_robots/__main__.py:26``):
    Usage: strands-robots <command> [options]

But the empty-argv branch at ``strands_robots/__main__.py:18`` still prints:
    Usage: python -m strands_robots <command>

harness#475 fixed the ``-h/--help/-V/--version`` branches ("makes a fresh
install look broken") but missed the no-args branch - the one a user hits
most often when they forget which subcommand they want.

Run:
    python bugbash_repros/main_no_args_usage_wrong_binary_name_repro.py

Prints a side-by-side table showing which invocation the user TYPED and
which invocation the usage line RECOMMENDS. Exit 0 when both agree,
1 on the drift.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Return (exit, stdout, stderr) for one subprocess call."""
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
    return r.returncode, r.stdout, r.stderr


def main() -> int:
    cases = [
        # (label, argv, expected invocation name in the usage line)
        ("strands-robots (console script, no args)",
         ["strands-robots"],
         "strands-robots"),
        ("strands-robots --help (console script)",
         ["strands-robots", "--help"],
         "strands-robots"),
        ("python -m strands_robots (module, no args)",
         [sys.executable, "-m", "strands_robots"],
         "python -m strands_robots"),
    ]

    print(f"{'invocation':<48} {'exit':<5} {'first-line Usage says':<42} {'drift?':<6}")
    print("-" * 110)
    drift = 0
    for label, argv, expected in cases:
        rc, out, err = _run(argv)
        first = out.splitlines()[0] if out else "(empty)"
        is_drift = expected not in first
        marker = "DRIFT" if is_drift else "ok"
        if is_drift:
            drift += 1
        print(f"{label:<48} {rc:<5} {first:<42} {marker:<6}")

    print()
    if drift:
        print(f"FAIL: {drift} case(s) recommend a different invocation than the one typed.")
        print("  See docs/reference/cli.md:7 - the console-script name is the documented one;")
        print("  strands_robots/__main__.py:18 still names 'python -m strands_robots'.")
        return 1
    print("PASS: every usage line names the invocation the user typed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
