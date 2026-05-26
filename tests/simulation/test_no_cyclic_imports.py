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
import re
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

# Traceback-shape regex for the regression assertions below. We anchor on
# the start-of-line ``ExceptionName:`` framing that Python emits for an
# uncaught traceback (and for chained ``raise from`` re-raises) rather than
# substring-matching the bare exception name. Substring matching is too
# loose: any benign log line, deprecation warning, or docstring snippet
# that happens to mention ``ImportError`` / ``RecursionError`` (for
# example, an optional-dep warning saying "...will raise ImportError on
# Python 3.13...") would otherwise fail the pin even though the import
# itself succeeded. The traceback frame is the only stderr shape that
# *actually* indicates the failure mode this pin exists to catch.
_TRACEBACK_EXC_PATTERN = re.compile(r"^(ImportError|RecursionError):", re.MULTILINE)


def _subprocess_env() -> dict[str, str]:
    """Env that keeps the parent ``sys.path`` visible to the child.

    A bare ``python -c "import ..."`` in CI relies on the parent process's
    ``sys.path`` (set by the editable install or the wheel install on the
    runner). Rather than re-deriving that path, propagate it via
    ``PYTHONPATH``: same module-resolution behaviour as the parent, but a
    fresh ``sys.modules`` cache so the per-process import-time dynamics
    are exercised honestly.

    Empty entries in ``sys.path`` and an empty inherited ``PYTHONPATH`` are
    filtered out before joining. POSIX interprets a leading, trailing, or
    interior empty pathsep entry as the *current working directory*, which
    on a CI runner is the checkout root: that would silently inject ``.``
    into the child's ``sys.path`` and could mask a regression where a
    module is only importable because ``cwd`` happens to contain it.
    """
    env = os.environ.copy()
    # Build the explicit path list: real entries from sys.path first, then
    # the parent's PYTHONPATH if (and only if) it is set and non-empty.
    # ``-I`` would strip these and produce a false positive.
    parts = [p for p in sys.path if p]
    inherited = env.get("PYTHONPATH", "")
    if inherited:
        # Split on pathsep and filter empty entries individually rather
        # than appending the inherited string verbatim. POSIX interprets
        # any empty pathsep entry (leading ``:foo``, trailing ``foo:``,
        # interior ``foo::bar``) as the *current working directory*, so
        # appending an inherited PYTHONPATH that already contains an
        # internal empty would silently inject ``cwd`` into the child's
        # ``sys.path`` even though we filter our own ``sys.path`` empties
        # above.
        parts.extend(p for p in inherited.split(os.pathsep) if p)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


@pytest.mark.parametrize("module", _SIMULATION_TRIPLE)
def test_module_imports_in_fresh_interpreter(module: str) -> None:
    """Each module imports cleanly in a brand-new Python process.

    Spawns ``python -c "import <module>"`` so the import is not contaminated
    by the test runner's already-cached ``sys.modules``. Asserts exit code 0
    and no ``RecursionError`` / ``ImportError`` traceback frames in stderr;
    on failure surfaces both for diagnosis.

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
    # An import that "succeeds" (exit 0) but emits a traceback on stderr
    # from a swallowed inner frame is still a regression. The regex anchors
    # on the start-of-line ``ExceptionName:`` framing so benign log lines
    # that happen to mention these exception names by name do not fail
    # the pin.
    match = _TRACEBACK_EXC_PATTERN.search(result.stderr)
    assert match is None, (
        f"{match.group(1)} traceback surfaced during fresh-interpreter import of {module}:\n{result.stderr}"
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
    match = _TRACEBACK_EXC_PATTERN.search(result.stderr)
    assert match is None, (
        f"{match.group(1)} traceback surfaced during combined fresh-interpreter "
        f"import of the simulation triple:\n{result.stderr}"
    )
