"""A process that joined the mesh must exit on its own.

``Robot("so100", mesh=True)`` puts two peers on the shared zenoh session: the
Simulation and a child peer for the SimRobot. zenoh serves each subscriber
callback from a non-daemon thread that lives until the session closes. The
session's exit hook therefore has to run *before* ``threading._shutdown()``
joins those threads - an ``atexit`` hook runs after and never gets its turn,
so the documented script (which never calls ``stop()``) hung forever, and so
did one that only stopped the root peer (the child still held the session).

The package registers two interpreter-exit teardowns for the mesh: this
module's session singleton, and
:func:`~strands_robots.tools.robot_mesh._stop_gateway_mesh` for the robot-less
gateway. Only the first owns the session, and closing the session is what ends
zenoh's threads - so the gateway process is pinned here rather than at its own
site, because what lets it exit is this hook and not its own. Every thread the
gateway's ``Mesh.stop`` ends is a daemon, which the interpreter never waits on.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from strands_robots.mesh import session as mesh_session

pytest.importorskip("zenoh")
pytest.importorskip("mujoco")

_DEADLINE_S = 30.0

# tests/conftest.py exports STRANDS_MESH=false for the suite; these subprocesses
# opt in the way the docs tell the reader to.
_MESH_ENV = {"STRANDS_MESH": "true", "STRANDS_MESH_LOCAL_DEV": "true"}


def _child_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """The environment a mesh subprocess under test runs in.

    The child inherits ``os.environ``, so without a redirect it resolves the
    audit log to the host's own ``~/.strands_robots/mesh_audit.jsonl`` - and a
    real mesh writes there: the gateway's ``peers`` records an
    ``llm_tool_action`` row, and a joining peer emits event-driven rows. That
    log is append-only and tamper-evident, so a suite-written record cannot be
    removed afterwards, and every unsigned one reads as forgery the moment an
    operator sets a PSK (AGENTS.md rule 16). Redirecting to a per-test root is
    what the rest of ``tests/mesh/`` does, and rule 15 is why it is derived
    from ``tmp_path`` rather than spelled as a fixed path.

    Args:
        tmp_path: The test's own scratch directory.
        **extra: Further variables for this one child.

    Returns:
        The overlay to apply on top of the current environment.
    """
    return {**_MESH_ENV, "STRANDS_MESH_AUDIT_DIR": str(tmp_path / "audit"), **extra}


_DOC_EXAMPLE = """
from strands_robots import Robot
sim_a = Robot("so100", mesh=True)
print(sim_a.mesh.peers)
"""

# A robot-less process reaching the fleet through the agent tool: the branch
# that builds the cached gateway Mesh, whose own hook cannot run until the
# session this module owns has already released zenoh's threads.
_GATEWAY_EXAMPLE = """
from strands_robots.tools import robot_mesh
print(robot_mesh(action="peers")["status"])
"""


def _seconds_to_exit(script: str, env: dict[str, str]) -> float:
    """Run ``script`` to completion and return how long it took to exit.

    Args:
        script: Python source to run in a fresh interpreter.
        env: Environment overlaid on the current one.

    Returns:
        Wall-clock seconds from spawn to exit.
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=_DEADLINE_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"mesh process did not exit within {_DEADLINE_S:.0f} s")
    assert proc.returncode == 0, proc.stderr[-2000:]
    return time.monotonic() - t0


def _registered_callables(hooks: list) -> list:
    """Every callable reachable through threading's exit-hook wrappers.

    CPython wraps each registration rather than storing the callable, and the
    wrapper shape is not part of any contract: 3.13 appends
    ``lambda: func(*arg, **kwargs)``, so the target is a closure cell and not a
    ``functools.partial.func``. Both are unwrapped here so this pin survives
    either shape.

    Args:
        hooks: Contents of ``threading._threading_atexits``.

    Returns:
        The wrappers and the callables they close over.
    """
    found = []
    for hook in hooks:
        found.append(hook)
        target = getattr(hook, "func", None)  # functools.partial
        if target is not None:
            found.append(target)
        for cell in getattr(hook, "__closure__", None) or ():  # lambda closure
            try:
                found.append(cell.cell_contents)
            except ValueError:  # an empty cell holds nothing to compare
                continue
    return found


def test_the_session_teardown_uses_the_pre_join_hook():
    """A fast, legible failure for the regression the deadlines below catch slowly."""
    # concurrent.futures registers its worker shutdown the same way for the
    # same reason; both must be in the pre-join hook list.
    hooks = getattr(threading, "_threading_atexits", None)
    assert hooks is not None, "threading._register_atexit vanished; fall back to atexit and re-measure"
    assert mesh_session._register_shutdown_hook is threading._register_atexit
    # And not (only) on atexit, where it could never run while a peer is alive.
    assert mesh_session._register_shutdown_hook is not atexit.register
    assert mesh_session._atexit_cleanup in _registered_callables(hooks)


@pytest.mark.parametrize("tail", ["", "sim_a.mesh.stop()"])
def test_documented_example_exits(tail: str, tmp_path: Path):
    assert _seconds_to_exit(_DOC_EXAMPLE + tail, _child_env(tmp_path)) < _DEADLINE_S


def test_a_robot_less_gateway_process_exits(tmp_path: Path):
    """The second exit hook's process, which the session close is what frees."""
    # The gateway waits one heartbeat period for presence on first bring-up;
    # this asks about exiting, not about discovery, so skip the wait.
    env = _child_env(tmp_path, STRANDS_MESH_GATEWAY_DISCOVERY_WAIT_S="0")
    assert _seconds_to_exit(_GATEWAY_EXAMPLE, env) < _DEADLINE_S
