"""Regression: no comparison may be written between two literal constants.

``assert True > 0`` reads as a measurement and is not one. Both operands are
literals, so the result is fixed when the line is typed: the assertion cannot
fail, cannot notice a change in what it claims to measure, and cannot tell a
correct premise from a mis-transcribed one. The shape turns up where it does the
most damage -- in a *premise* test, written to record why some guard is needed,
whose whole value is that the claim was measured rather than restated.

Neither merge gate covers it.

``ruff`` selects ``B015`` (useless-comparison) for exactly this capability:
``.github/codeql/codeql-config.yml`` hands it that job in writing, because ruff
is merge-blocking here where CodeQL is advisory. But B015 fires only when a
comparison's *result is unused*, so it is silent by design on every instance
inside an ``assert`` -- the ``assert`` consumes the result. Measured on the two
sites this guard was written for, ``ruff check`` reports no finding at all.

CodeQL's ``py/comparison-of-constants`` is unreliable across instances. It
reported ``True < 1`` on a pull-request ref while the twin ``True > 0`` sat on
``main`` unreported, through a default-branch analysis that was both current and
successful.

So the shape is refused here instead: deterministically, on every comparison in
the repository rather than on whichever one a scanner happens to surface, in the
gate that blocks a merge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots

# The repository root, reached through this module's own location rather than a
# path literal.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every tree whose Python ships with the repository. The package is reached
# through the imported module so a layout change cannot silently narrow the scan
# to nothing, and the remaining roots are asserted to exist below.
_SCAN_ROOTS = (
    Path(strands_robots.__file__).resolve().parent,
    _REPO_ROOT / "tests",
    _REPO_ROOT / "tests_integ",
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "examples",
)


def _python_sources() -> list[Path]:
    return sorted(p for root in _SCAN_ROOTS for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def constant_comparisons(source: str) -> list[tuple[int, str]]:
    """Comparisons in ``source`` whose every operand is a literal.

    A tuple, an f-string and a name are all excluded: only a comparison the
    reader can evaluate without running anything is reported, which is the one
    shape that has no legitimate use.

    Args:
        source: Python source text.

    Returns:
        One ``(line number, expression)`` pair per comparison whose result is
        decided before it runs.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if all(isinstance(operand, ast.Constant) for operand in operands):
                found.append((node.lineno, ast.unparse(node)))
    return found


def test_scan_roots_discovered() -> None:
    """Guard: the scan walked the whole repository rather than one subtree."""
    for root in _SCAN_ROOTS:
        assert root.is_dir(), f"scan root missing: {root}"
    sources = _python_sources()
    assert len(sources) > 1000
    top_level = {p.relative_to(_REPO_ROOT).parts[0] for p in sources}
    assert {"strands_robots", "tests", "tests_integ", "scripts", "examples"} <= top_level


def test_no_comparison_is_decided_before_it_runs() -> None:
    """No shipped comparison has a literal on both sides."""
    offenders: list[str] = []
    for path in _python_sources():
        rel = path.relative_to(_REPO_ROOT)
        for line, expression in constant_comparisons(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{line}  {expression}")
    assert not offenders, (
        "Comparisons whose result is fixed when they are typed, so they measure nothing:\n  " + "\n  ".join(offenders)
    )


def test_scanner_detects_a_planted_constant_comparison() -> None:
    """Meta: an empty result means clean sources, not a scanner that matches nothing."""
    planted = "def f() -> None:\n    assert True > 0\n"
    assert constant_comparisons(planted) == [(2, "True > 0")]


def test_scanner_accepts_every_comparison_with_a_runtime_operand() -> None:
    """A comparison that is actually decided when it runs is not flagged."""
    legitimate = (
        "LIMIT = 1\n"
        "def f(value: float) -> None:\n"
        "    assert value > 0\n"
        "    assert LIMIT < 2\n"
        "    assert len(str(value)) > 0\n"
        "    assert (1, 2) == (1, 2)\n"
        "    assert f'{value}' == '1'\n"
    )
    assert constant_comparisons(legitimate) == []
