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
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

import strands_robots
from tests._package_ast import parse_file, parse_source, walk_tree

_TESTS = Path(__file__).resolve().parent

#: What a grader reads a file's source with, inline or into a name it then parses.
_READ = "read_text"


_SCOPES = ast.FunctionDef | ast.AsyncFunctionDef


def _reads_a_file(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == _READ for n in ast.walk(node))


def _bound_by_a_read(statements: Iterable[ast.AST]) -> set[str]:
    return {
        target.id
        for node in statements
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None and _reads_a_file(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }


def _parse_calls(nodes: Iterable[ast.AST], names: set[str]) -> Iterator[int]:
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast"
        ):
            source = node.args[0]
            if source.id in names if isinstance(source, ast.Name) else _reads_a_file(source):
                yield node.lineno


def fresh_parses(tree: ast.Module) -> list[int]:
    """Lines where ``ast.parse`` is handed a source the module read rather than wrote.

    Three spellings hand it one: the read inline (``ast.parse(path.read_text())``),
    a name bound from a read - in the same function or at module level, unless
    the function rebinds it - and a parameter, which is how a helper that a
    planted negative can also call is handed a file's text. Any of them is a
    shared file's text as often as not, and :func:`~tests._package_ast.parse_source`
    returns that file's shared tree and parses any other text afresh, so it is the
    drop-in for all three.

    Args:
        tree: A parsed test module.

    Returns:
        The sorted line numbers of the offending calls.
    """
    module_reads = _bound_by_a_read(tree.body)
    top_level = [n for node in tree.body if not isinstance(node, _SCOPES | ast.ClassDef) for n in ast.walk(node)]
    found = set(_parse_calls(top_level, module_reads))
    for function in (n for n in walk_tree(tree) if isinstance(n, _SCOPES)):
        nodes = list(ast.walk(function))
        args = function.args
        # A test's own parameters are fixtures and parametrized snippets, never a file it read.
        is_test = function.name.startswith("test")
        params = set() if is_test else {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        rebound = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        found.update(_parse_calls(nodes, params | _bound_by_a_read(nodes) | (module_reads - rebound)))
    return sorted(found)


def test_no_test_module_parses_a_source_it_read_afresh() -> None:
    offenders = sorted(
        f"{path.relative_to(_TESTS).as_posix()}:{line}"
        for path in _TESTS.rglob("*.py")
        if path.name != "_package_ast.py" and "ast.parse(" in path.read_text(encoding="utf-8")
        for line in fresh_parses(parse_file(path))
    )
    assert offenders == [], f"use tests._package_ast.parse_source(text) instead: {offenders}"


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ('tree = ast.parse(path.read_text(encoding="utf-8"))', [1]),
        ("tree = ast.parse((ROOT / rel).read_text(), filename=rel)", [1]),
        ("def f(path):\n    text = path.read_text()\n    return ast.parse(text)", [3]),
        ("SOURCE = PATH.read_text()\n\n\ndef f():\n    return ast.parse(SOURCE)", [5]),
        ("def f(source: str):\n    return ast.parse(source)", [2]),
        ("tree = parse_source(path.read_text())", []),
        ("tree = ast.parse('x = 1')", []),
        ("def f():\n    snippet = 'x = 1'\n    return ast.parse(snippet)", []),
        ("def f():\n    return ast.parse(textwrap.dedent(inspect.getsource(f)))", []),
    ],
    ids=[
        "inline",
        "inline-built-path",
        "local",
        "module-name",
        "parameter",
        "shared",
        "literal",
        "literal-local",
        "fn",
    ],
)
def test_the_rule_reads_every_spelling_that_hands_ast_parse_a_read(source: str, lines: list[int]) -> None:
    assert fresh_parses(ast.parse(source)) == lines


#: ``tomllib.load(s)`` of ``uv.lock`` re-reads 1.5 MB of TOML, which came to 70
#: parses in one process across the packaging rules.
_FRESH_LOCK_READ = re.compile(r"tomllib\.loads?\([^\n]*(?:LOCK|uv\.lock)")


def test_no_test_module_reads_uv_lock_afresh() -> None:
    offenders = sorted(
        path.relative_to(_TESTS).as_posix()
        for path in _TESTS.rglob("*.py")
        if path.name != "uv_lock_closure.py"
        and path != Path(__file__)
        and _FRESH_LOCK_READ.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], f"use tests.uv_lock_closure.uv_lock() instead: {offenders}"


@pytest.mark.parametrize(
    ("line", "fresh"),
    [
        ('lock = tomllib.loads(_LOCK.read_text(encoding="utf-8"))', True),
        ('lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())', True),
        ("lock = tomllib.load(_UV_LOCK.open('rb'))", True),
        ('data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))', False),
    ],
)
def test_the_lock_read_is_recognised_however_the_path_is_built(line: str, fresh: bool) -> None:
    assert bool(_FRESH_LOCK_READ.search(line)) is fresh


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
