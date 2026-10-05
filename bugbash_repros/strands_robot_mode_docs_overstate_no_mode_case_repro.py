#!/usr/bin/env python3
"""
Repro: STRANDS_ROBOT_MODE env var is advertised as deciding mode when the caller
passes no `mode`, but the Robot() factory's hardcoded `mode: str = "sim"` default
wins first and the env var is only consulted on the explicit `mode="auto"` branch.

Target: strands-labs/robots @ main (972df275c)
Rotation: readme_quickstart (end-user Install -> Quickstart surface)
Defect class: Docs mismatch (two concept pages overstate env-var override)

The claim appears in TWO places:

  docs/concepts/robots.md:18
    "With no `mode`, `STRANDS_ROBOT_MODE` decides, then a hardware probe, then a
     USB scan, and sim is the fallback"

  docs/concepts/architecture.md:30
    "`STRANDS_ROBOT_MODE` overrides `mode` from the environment."

The code at strands_robots/robot.py:632 has:

    def Robot(name: str, mode: str = "sim", ...)

and strands_robots/robot.py:805:

    if mode == "auto":
        mode = _auto_detect_mode(canonical)

So `STRANDS_ROBOT_MODE` only takes effect when the user explicitly writes
``mode="auto"``.  "No mode" lands on the ``mode="sim"`` default, which short-
circuits the env-var read entirely.

This is NOT a safety hole (sim-default is the safe answer), but a reader of
either doc page who sets `STRANDS_ROBOT_MODE=real` and then runs
`Robot("so100")` expecting hardware control gets a quiet sim and no warning.
The function docstring (strands_robots/robot.py:93) ALSO lists the env var as
priority #1 "explicit override", reinforcing the misread.

Run:
    SYSTEM_PROMPT= MUJOCO_GL=egl python strands_robot_mode_docs_overstate_no_mode_case_repro.py

Exits 0 and prints the mismatch rows; exits 1 if any row contradicts the
measurement this repro pins (acts as a self-check if upstream fixes the gap).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _run_case(env_mode: str | None, mode_kwarg: str | None) -> tuple[str, str]:
    """Run `Robot("so100", **kwarg)` with STRANDS_ROBOT_MODE=env_mode in a child.

    Returns (type_name, stderr_tail).  Keeps SYSTEM_PROMPT stripped (Thor
    contract) and inserts the local checkout on sys.path so this script works
    whether or not strands-robots is pip-installed.
    """
    arg = "" if mode_kwarg is None else f", mode={mode_kwarg!r}"
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        'os.environ.pop("SYSTEM_PROMPT", None)\n'
        + (
            f'os.environ["STRANDS_ROBOT_MODE"] = {env_mode!r}\n'
            if env_mode is not None
            else 'os.environ.pop("STRANDS_ROBOT_MODE", None)\n'
        )
        + "from strands_robots import Robot\n"
        "try:\n"
        f"    r = Robot('so100'{arg})\n"
        "    print('TYPE:', type(r).__name__)\n"
        "except Exception as e:\n"
        "    print('ERR:', type(e).__name__, str(e)[:160])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    env["MUJOCO_GL"] = "egl"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc.stdout.strip(), proc.stderr.strip()


def main() -> int:
    print("=" * 78)
    print("REPRO: STRANDS_ROBOT_MODE docs overstate env-var override")
    print("=" * 78)
    print()

    # Pinned measurement: three rows that together show the gap.
    #
    # Row A -- "no mode" with env=real: docs claim env decides.  Measured: sim.
    # Row B -- mode='auto' with env=real: env DOES drive detection (expected).
    # Row C -- mode='sim' with env=real: env ignored (sane; override only in auto).
    cases = [
        ("A", None, "real", "Robot('so100')              # 'no mode' per docs"),
        ("B", "auto", "real", "Robot('so100', mode='auto')"),
        ("C", "sim", "real", "Robot('so100', mode='sim')"),
    ]

    rows = []
    for label, kwarg, env_mode, source in cases:
        out, err = _run_case(env_mode, kwarg)
        rows.append((label, env_mode, kwarg, out, err))
        print(f"[{label}] STRANDS_ROBOT_MODE={env_mode!r}  {source}")
        print(f"    -> {out}")
        if err and "ResourceWarning" not in err:
            print(f"    (stderr: {err[-160:]!r})")
        print()

    # Assertions (self-check).  If upstream fixes the docs OR flips the code to
    # honour the env on the sim default, this list is the first thing to update.
    expectations = {
        # (label): expected prefix of `out`
        "A": "TYPE: MuJoCoSimEngine",   # DOCS SAY REAL, ACTUAL IS SIM  <-- the gap
        "B": "ERR: ",                   # env=real takes effect, fails because no port
        "C": "TYPE: MuJoCoSimEngine",   # explicit sim always wins (sane)
    }

    ok = True
    print("-" * 78)
    print("Pinned expectations (what this repro proves):")
    for label, env_mode, kwarg, out, _ in rows:
        want = expectations[label]
        passed = out.startswith(want)
        marker = "OK" if passed else "DRIFT"
        print(f"  [{label}] {marker}  wanted startswith {want!r}")
        if not passed:
            ok = False

    print()
    print("Gap: docs/concepts/robots.md:18 and docs/concepts/architecture.md:30")
    print("say STRANDS_ROBOT_MODE decides when `no mode` is passed.")
    print("Row [A] is 'no mode' -> MuJoCoSimEngine, so the env var did NOT decide.")
    print()
    print("Not a safety hole (sim-default is safe); fix is a one-sentence doc")
    print("rewrite scoping the env-var's role to the mode='auto' branch, where")
    print("strands_robots/robot.py:103-107 actually reads it.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
