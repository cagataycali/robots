"""A package or test source file is parsed once per test process, not once per grader.

About 170 graders walk ``strands_robots`` and parse every file. Each parsed on
its own, many at module scope, which pytest-xdist repeats on every worker at
collection. Measured on the whole-tree roster at 14 workers, that was 134,919
parses of package files costing 775 worker-seconds, against a floor of 31
seconds for parsing each file once per worker. About 30 more graders walk
``tests/`` the same way. :func:`tests._package_ast.parse_file` is that floor
for both trees; this module keeps the tree from drifting back, and holds ``uv.lock`` to
:func:`tests.uv_lock_closure.uv_lock` the same way.
"""

from __future__ import annotations

import ast
import copy
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import strands_robots
from tests._package_ast import parse_file, parse_source, walk_tree

_TESTS = Path(__file__).resolve().parent

#: Each fresh-parse spelling, the one module allowed to spell it, and what to call
#: instead. ``ast.parse(<expression>.read_text(...))`` on one line, whatever builds
#: the path (``Path(mod.__file__)``, ``(ROOT / rel)``), parses a source afresh
#: wherever it is; ``tomllib.load(s)`` of ``uv.lock`` re-reads 1.5 MB of TOML, which
#: came to 70 parses in one process across the packaging rules.
_FRESH_PARSES = (
    (re.compile(r"ast\.parse\([^\n]*\.read_text\("), "_package_ast.py", "tests._package_ast.parse_file(path)"),
    (re.compile(r"tomllib\.loads?\([^\n]*(?:LOCK|uv\.lock)"), "uv_lock_closure.py", "tests.uv_lock_closure.uv_lock()"),
)


@pytest.mark.parametrize(("spelling", "owner", "remedy"), _FRESH_PARSES, ids=["source", "uv-lock"])
def test_no_test_module_parses_a_shared_file_afresh(spelling: re.Pattern[str], owner: str, remedy: str) -> None:
    offenders = sorted(
        path.relative_to(_TESTS).as_posix()
        for path in _TESTS.rglob("*.py")
        if path.name != owner and path != Path(__file__) and spelling.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], f"use {remedy} instead: {offenders}"


#: One file of each tree whose parse is shared: the package and the test tree.
_GRADED_FILES = (Path(strands_robots.__file__), Path(__file__))


def test_a_graded_file_is_parsed_once_and_any_other_file_every_time(tmp_path: Path) -> None:
    for graded in _GRADED_FILES:
        assert parse_file(graded) is parse_file(graded), graded

    fixture = tmp_path / "fixture.py"
    fixture.write_text("x = 1\n", encoding="utf-8")
    first = parse_file(fixture)
    fixture.write_text("y = 2\n", encoding="utf-8")
    assert [t.id for t in first.body[0].targets] == ["x"]  # type: ignore[attr-defined]
    assert [t.id for t in parse_file(fixture).body[0].targets] == ["y"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("row", "line", "fresh"),
    [
        (0, 'tree = ast.parse(path.read_text(encoding="utf-8"))', True),
        (0, "tree = ast.parse(Path(mod.__file__).read_text())", True),
        (0, "tree = ast.parse((ROOT / rel).read_text(), filename=rel)", True),
        (0, "tree = parse_file(ROOT / rel)", False),
        (0, "tree = ast.parse(source)", False),
        (1, 'lock = tomllib.loads(_LOCK.read_text(encoding="utf-8"))', True),
        (1, 'lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())', True),
        (1, "lock = tomllib.load(_UV_LOCK.open('rb'))", True),
        (1, 'data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))', False),
    ],
)
def test_the_fresh_parse_spelling_is_recognised_however_the_path_is_built(row: int, line: str, fresh: bool) -> None:
    assert bool(_FRESH_PARSES[row][0].search(line)) is fresh


def test_a_graded_files_text_is_parsed_once_and_any_other_text_every_time() -> None:
    for graded in _GRADED_FILES:
        assert parse_source(graded.read_text(encoding="utf-8")) is parse_file(graded), graded
    assert parse_source("x = 1\n") is not parse_source("x = 1\n")


def test_a_shared_tree_is_walked_once_and_any_other_node_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """``walk_tree`` replays a shared tree's nodes in ``ast.walk`` order, and walks anything else afresh."""
    shared = parse_file(Path(__file__))
    nodes = list(walk_tree(shared))
    assert nodes == list(ast.walk(shared))
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef))
    edited = copy.deepcopy(shared)
    edited.body.append(ast.Pass())
    expected = [len(list(ast.walk(function))), len(nodes) + 1]

    walks: list[ast.AST] = []
    real_walk = ast.walk

    def counting_walk(node: ast.AST) -> Iterator[ast.AST]:
        walks.append(node)
        return real_walk(node)

    monkeypatch.setattr(ast, "walk", counting_walk)
    assert list(walk_tree(shared)) == nodes
    assert [len(list(walk_tree(function))), len(list(walk_tree(edited)))] == expected
    assert walks == [function, edited], "the shared tree was walked again"
