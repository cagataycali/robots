"""Regression test for unsafe cyclic imports in strands_robots.simulation.

Pinned by review feedback addressing CodeQL alerts #83, #84, #85, #86, #87
(``py/unsafe-cyclic-import`` errors). Each module in the simulation package
must import cleanly in any order, regardless of which module is imported
first by a fresh interpreter. Prior to the fix, importing
``strands_robots.simulation.policy_runner`` before
``strands_robots.simulation.base`` could leave ``SimEngine`` undefined at
``base.py``'s top level because of a cycle through a runtime
``from strands_robots.simulation.policy_runner import PolicyRunner,
VideoConfig`` statement at module scope.

Each subprocess run starts a fresh interpreter so the import order under
test is the *primary* cause of failure rather than benefiting from
side-effects of earlier imports inside this test process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_MODULES = [
    "strands_robots.simulation",
    "strands_robots.simulation.base",
    "strands_robots.simulation.benchmark",
    "strands_robots.simulation.policy_runner",
]


@pytest.mark.parametrize("module", _MODULES)
def test_module_imports_in_fresh_interpreter(module: str) -> None:
    """Each simulation module must import cleanly when it is the first
    module pulled in by a fresh interpreter.

    Regression for CodeQL alerts #83, #84, #85, #86, #87
    (``py/unsafe-cyclic-import``). The cycle was
    ``simulation.base`` -> ``simulation.policy_runner`` ->
    ``simulation.base`` (via TYPE_CHECKING + module-level runtime imports).
    A static analyser cannot prove the runtime path is safe in all import
    orderings, so the safe fix is to defer the cycle-closing import to
    method scope.
    """

    code = f"import {module}; print('OK')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Importing {module} in a fresh interpreter failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
