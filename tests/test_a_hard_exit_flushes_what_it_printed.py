"""A script that ends in ``os._exit`` flushes the report it printed first.

``os._exit`` is the right ending for a process whose teardown can crash - an
Isaac ``SimulationApp``, a Tegra EGL GL destructor, a wedged executor thread -
because it runs no ``atexit`` hook, no ``__del__`` and no ``finally`` block.
That is also what makes it a reporting hazard: CPython's ``sys.stdout`` is
block-buffered whenever it is not a tty, so every ``print`` still sitting in
that buffer is discarded with the rest of the teardown.

Measured on ``74136572a``, before the call site this module guards was paired.
``examples/vla/cosmos3_diffusers_mujoco_rollout.py`` printed the Cosmos action
chunk shape and the Cartesian tracking error - the two numbers the script exists
to report - and then hard-exited. Run on a real ``nvidia/Cosmos3-Nano`` forward
pass with stdout redirected to a file::

    $ python examples/vla/cosmos3_diffusers_mujoco_rollout.py --render out.mp4 > run.log
    $ grep -c 'tracking error' run.log
    0

The video was written and the process exited 0, so nothing looked wrong; the
whole measurement was simply gone. The same script crashing before the hard exit
(a missing distribution) showed its earlier print, because that exit path is
CPython's, which flushes.

The rule is per call site rather than per file: only the code around a given
exit knows what it has printed. A ``sys.stdout.flush()`` before the exit
satisfies it, and so does a ``print(..., flush=True)``, which flushes the whole
buffer including any plain ``print`` that preceded it - that is the form
``strands_robots/robot.py`` uses for its two exits.

``tests/`` is out of scope: a test that hard-exits is failing the run anyway, and
pytest captures its output through a replaced stream rather than the process's.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCANNED_TREES = ("examples", "strands_robots")


def _hard_exit_sites(tree: ast.AST) -> list[ast.Call]:
    """Every ``os._exit(...)`` call in a parsed module."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]


def _flush_lines(scope: ast.AST) -> list[int]:
    """Lines in ``scope`` that flush stdout, either explicitly or via ``print``.

    Line numbers are the ordering used because a flush is only useful before the
    exit, and an exit is terminal: no loop carries control back above it.
    """
    lines = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "flush":
            lines.append(node.lineno)
        elif isinstance(node.func, ast.Name) and node.func.id == "print":
            if any(keyword.arg == "flush" for keyword in node.keywords):
                lines.append(node.lineno)
    return lines


def _unflushed_exits(path: Path) -> list[int]:
    """Line numbers of every hard exit in ``path`` with no flush above it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scopes: list[ast.AST] = [tree] + [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    offenders = []
    for scope in scopes:
        flushes = _flush_lines(scope)
        for call in _hard_exit_sites(scope):
            if not any(line < call.lineno for line in flushes):
                offenders.append(call.lineno)
    return sorted(set(offenders))


def _scanned_sources() -> list[Path]:
    return [
        path
        for tree_name in _SCANNED_TREES
        for path in sorted((_REPO_ROOT / tree_name).rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def test_every_hard_exit_flushes_stdout_first() -> None:
    """No shipped ``os._exit`` discards what its own run printed."""
    offenders = [
        f"{path.relative_to(_REPO_ROOT).as_posix()}:{lineno}"
        for path in _scanned_sources()
        for lineno in _unflushed_exits(path)
    ]
    assert not offenders, (
        "os._exit runs no atexit hook, so buffered stdout is dropped: redirected "
        "to a file, the run's own report never lands. Flush before exiting "
        "(sys.stdout.flush(), or print(..., flush=True)): " + ", ".join(offenders)
    )


def test_the_scan_reaches_the_exits_it_grades() -> None:
    """Non-vacuity: hard exits exist in the scanned trees and are being found."""
    found = [
        f"{path.relative_to(_REPO_ROOT).as_posix()}:{call.lineno}"
        for path in _scanned_sources()
        for call in _hard_exit_sites(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]
    assert found, f"no os._exit call found under {list(_SCANNED_TREES)}; the scan is not reaching the code"


def test_the_guard_separates_a_flushed_exit_from_a_bare_one(tmp_path: Path) -> None:
    """Planted positive: each accepted form passes and the bare form fails."""
    cases = {
        "bare.py": "import os\nprint('result')\nos._exit(0)\n",
        "explicit.py": "import os, sys\nprint('result')\nsys.stdout.flush()\nos._exit(0)\n",
        "flushing_print.py": "import os\nprint('result', flush=True)\nos._exit(0)\n",
        "in_function.py": "import os\ndef main():\n    print('result')\n    os._exit(0)\n",
    }
    for name, source in cases.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    verdicts = {name: bool(_unflushed_exits(tmp_path / name)) for name in cases}
    assert verdicts == {
        "bare.py": True,
        "explicit.py": False,
        "flushing_print.py": False,
        "in_function.py": True,
    }, verdicts
