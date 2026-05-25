"""Regression: simulation triple imports cleanly in fresh interpreters.

Complement to ``tests/simulation/test_no_import_cycle.py``. Where that test
asserts the *static* runtime-import graph (parsed from source) is acyclic,
this test asserts the *dynamic* import actually succeeds in a clean Python
process for each affected module.

The two pins catch different failure modes:

- A static-graph regression (e.g. someone hoists ``from .. import SimEngine``
  out of a ``TYPE_CHECKING`` block in ``policy_runner.py``) is caught by
  ``test_no_import_cycle.py``: the AST scan would re-add the ``policy_runner
  -> base`` edge and the cycle would resurface.
- A dynamic-import regression (e.g. a top-level statement in one of the
  modules raises during import, or the import order produces a partial-module
  ``ImportError`` only at process start) is caught here: the static graph
  could remain acyclic while the *actual* import still fails. A subprocess
  per module isolates each import from cached state in the test runner.

Together they form the CodeQL-independent regression contract referenced
from ``.github/codeql/config.yml`` and ``.github/codeql/README.md``.

Pinned modules (the documented static-only cycle suppressed by
``py/unsafe-cyclic-import``):

- ``strands_robots.simulation.base``
- ``strands_robots.simulation.policy_runner``
- ``strands_robots.simulation.benchmark``
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# The simulation triple that participates in the documented static-only
# cycle. If a future suppression covers a new file shape, add it here so
# the regression contract grows with the suppression list.
_SIMULATION_TRIPLE: tuple[str, ...] = (
    "strands_robots.simulation.base",
    "strands_robots.simulation.policy_runner",
    "strands_robots.simulation.benchmark",
)


def _subprocess_env() -> dict[str, str]:
    """Env that keeps the parent ``sys.path`` visible to the child.

    A bare ``python -c "import ..."`` in CI relies on the parent process's
    ``sys.path`` (set by the editable install or the wheel install on the
    runner). Rather than re-deriving that path, propagate it via
    ``PYTHONPATH``: same module-resolution behaviour as the parent, but a
    fresh ``sys.modules`` cache so the per-process import-time dynamics
    are exercised honestly.
    """
    env = os.environ.copy()
    # Prepend the inherited sys.path so child resolves modules the same way
    # this test process does -- editable install, wheel install, or src/
    # checkout. ``-I`` would strip these and produce a false positive.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.parametrize("module", _SIMULATION_TRIPLE)
def test_module_imports_in_fresh_interpreter(module: str) -> None:
    """Each module imports cleanly in a brand-new Python process.

    Spawns ``python -c "import <module>"`` so the import is not contaminated
    by the test runner's already-cached ``sys.modules``. Asserts exit code 0
    and no ``RecursionError`` / ``ImportError`` traces in stderr; on failure
    surfaces both for diagnosis.

    The 30-second timeout guards against an infinite-recursion regression
    (the precise failure mode this pin exists to catch) hanging the suite.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"fresh-interpreter import of {module} failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Surface any import-time recursion or partial-failure traces as well.
    # An import that "succeeds" (exit 0) but emits a RecursionError on
    # stderr from a swallowed inner frame is still a regression.
    assert "RecursionError" not in result.stderr, (
        f"RecursionError surfaced during fresh-interpreter import of {module}:\n{result.stderr}"
    )
    assert "ImportError" not in result.stderr, (
        f"ImportError surfaced during fresh-interpreter import of {module}:\n{result.stderr}"
    )


def test_full_triple_imports_in_one_process() -> None:
    """Importing all three modules in a single process also succeeds.

    Catches the failure mode where each module imports cleanly in isolation
    (the per-module test above passes) but the *combination* hits a
    partial-module state -- e.g. ``base`` is mid-import, ``policy_runner``
    is imported as a side-effect, and ``policy_runner`` then re-enters
    ``base`` before its module dict is fully populated.
    """
    imports = "; ".join(f"import {m}" for m in _SIMULATION_TRIPLE)
    result = subprocess.run(
        [sys.executable, "-c", imports],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"combined fresh-interpreter import of the simulation triple failed "
        f"(exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
