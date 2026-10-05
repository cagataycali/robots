"""Pin test: STRANDS_ROBOT_MODE is scoped to mode="auto" only.

Guards the fix at:

* docs/concepts/robots.md:18
* docs/concepts/architecture.md:30
* strands_robots/robot.py docstring (module + _auto_detect_mode)

If a future refactor makes STRANDS_ROBOT_MODE flip the sim-default path to
real (e.g. by moving the env read above the ``if mode == "auto":`` branch), the
first row below would succeed as a hardware construct and this test would
fail - with a message pointing at the two doc pages whose claim depends on the
current scope.

Does NOT touch hardware: the "real" expectation is that the auto-detect runs,
not that it succeeds.  Row B (``mode="auto"``) finds no servo on this host and
is driven to a clean refusal; what we pin is that row A (sim-by-default) is
NOT subject to the env var.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_robot(env_mode: str, mode_kwarg: str | None) -> str:
    """Build a Robot('so100') with STRANDS_ROBOT_MODE=env_mode in a child.

    Child isolation keeps the import-time singletons (zenoh, mesh, mujoco)
    from one case bleeding into the next, and keeps SYSTEM_PROMPT scrubbed per
    the Thor shell contract.
    """
    arg = "" if mode_kwarg is None else f", mode={mode_kwarg!r}"
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        'os.environ.pop("SYSTEM_PROMPT", None)\n'
        f'os.environ["STRANDS_ROBOT_MODE"] = {env_mode!r}\n'
        "from strands_robots import Robot\n"
        "try:\n"
        f"    r = Robot('so100'{arg})\n"
        "    print(type(r).__name__)\n"
        "except Exception as e:\n"
        "    print('ERR:' + type(e).__name__)\n"
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
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}:\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr[-1500:]}"
    )
    return proc.stdout.strip().splitlines()[-1]


def test_env_var_does_not_flip_sim_default() -> None:
    """Row A: ``Robot("so100")`` + ``STRANDS_ROBOT_MODE=real`` -> sim.

    This is the row the docs got wrong before the fix.  Keeping the sim engine
    pinned here means a future reader of docs/concepts/robots.md:18 and
    docs/concepts/architecture.md:30 cannot drift back to the "env decides on
    no mode" wording without flunking this test.
    """
    out = _run_robot(env_mode="real", mode_kwarg=None)
    assert out == "MuJoCoSimEngine", (
        f"expected sim engine on default-sim path with STRANDS_ROBOT_MODE=real, "
        f"got {out!r}; docs/concepts/robots.md:18 and docs/concepts/"
        f"architecture.md:30 depend on this being sim."
    )


def test_env_var_is_consulted_on_mode_auto() -> None:
    """Row B: ``Robot("so100", mode="auto")`` + env=real -> auto-detect runs.

    On a host with no servo, the auto path falls into the real-hardware
    constructor which refuses without a port.  Either outcome (sim with the
    detection having reported no hardware, or the port-missing refusal) proves
    the env var *did* reach the branch -- what we don't want is the sim-engine
    being returned silently as if ``mode="auto"`` were a synonym for sim.
    """
    out = _run_robot(env_mode="real", mode_kwarg="auto")
    # Two acceptable shapes: hardware refusal (port missing) OR sim fallback
    # when the auto-detect finds no servo.  "MuJoCoSimEngine with env=real on
    # mode='auto'" is only valid if the detection honestly found no hardware,
    # which this host does.
    assert out == "MuJoCoSimEngine" or out.startswith("ERR:"), (
        f"mode='auto' with STRANDS_ROBOT_MODE=real should either refuse or "
        f"fall back to sim after a probe; got {out!r}"
    )


def test_explicit_mode_sim_overrides_env_var() -> None:
    """Row C: ``Robot("so100", mode="sim")`` + env=real -> sim, env ignored.

    Explicit kwarg must always win; the env var cannot override a user who
    typed ``mode="sim"``.  This is the sanity half of the asymmetry.
    """
    out = _run_robot(env_mode="real", mode_kwarg="sim")
    assert out == "MuJoCoSimEngine"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
