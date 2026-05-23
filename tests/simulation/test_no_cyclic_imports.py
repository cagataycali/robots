"""Import smoke test for strands_robots.simulation modules.

Each module in the simulation package must import cleanly in a fresh
interpreter, regardless of which module is imported first. This is a
smoke test that catches gross import failures (missing deps, syntax
errors, runtime exceptions at import time) but does NOT pin the
cyclic-import fix itself.

The actual regression pin for the CodeQL alerts #83-#87 cycle fix lives
in ``test_no_import_cycle.py::test_base_has_no_module_level_policy_runner_import``
which statically asserts that ``base.py`` never imports from
``strands_robots.simulation.policy_runner`` at module level.

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

    This is a smoke test — it catches gross import failures but does not
    pin the cyclic-import fix. The pin test lives in
    ``test_no_import_cycle.py::test_base_has_no_module_level_policy_runner_import``.
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
