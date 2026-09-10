"""The ``docs/mesh.md`` example must exit on its own.

``Robot("so100", mesh=True)`` puts two peers on the shared zenoh session: the
Simulation and a child peer for the SimRobot. zenoh serves each subscriber
callback from a non-daemon thread that lives until the session closes. The
session's exit hook therefore has to run *before* ``threading._shutdown()``
joins those threads - an ``atexit`` hook runs after and never gets its turn,
so the documented script (which never calls ``stop()``) hung forever, and so
did one that only stopped the root peer (the child still held the session).
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time

import pytest

from strands_robots.mesh import session as mesh_session

pytest.importorskip("zenoh")
pytest.importorskip("mujoco")

_DEADLINE_S = 30.0

_DOC_EXAMPLE = """
from strands_robots import Robot
sim_a = Robot("so100", mesh=True)
print(sim_a.mesh.peers)
"""


def test_exit_hook_is_registered_before_the_thread_join():
    # concurrent.futures registers its worker shutdown the same way for the
    # same reason; both must be in the pre-join hook list.
    hooks = getattr(threading, "_threading_atexits", None)
    assert hooks is not None, "threading._register_atexit vanished; fall back to atexit and re-measure"
    # The list holds functools.partial wrappers around the registered callables.
    assert mesh_session._atexit_cleanup in [getattr(h, "func", h) for h in hooks]
    # And not (only) on atexit, where it could never run while a peer is alive.
    assert mesh_session._register_shutdown_hook is not atexit.register


@pytest.mark.parametrize("tail", ["", "sim_a.mesh.stop()"])
def test_documented_example_exits(tail: str):
    # tests/conftest.py exports STRANDS_MESH=false for the suite; the example
    # opts in the way the page tells the reader to.
    env = {**os.environ, "STRANDS_MESH": "true", "STRANDS_MESH_LOCAL_DEV": "true"}
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DOC_EXAMPLE + tail],
            env=env,
            capture_output=True,
            text=True,
            timeout=_DEADLINE_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"documented mesh example did not exit within {_DEADLINE_S:.0f} s (tail={tail!r})")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert time.monotonic() - t0 < _DEADLINE_S
