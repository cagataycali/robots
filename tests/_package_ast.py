"""The parsed tree of a source file, shared by every grader in a process.

About 150 test modules grade the whole ``strands_robots`` tree by reading and
parsing every one of its ~380 files, and about 30 more grade the ~2,200 files
of ``tests/`` the same way. Each did that on its own, many at module scope,
which pytest-xdist runs once per worker during collection, so a run parsed
each tree dozens to hundreds of times - most of the time those graders spent.
:func:`parse_file` parses a file of the package, ``tests/`` or
``tests_integ/`` once per process and hands every later caller the same tree;
:func:`parse_source` does the same for a grader that is handed the file's text
instead of its path, and :func:`walk_tree` walks a shared tree once per process.

The tree is shared, so it is read-only by contract: a grader that edits one
(a planted negative, an ``ast.NodeTransformer``) works on
``copy.deepcopy(tree)``. A path outside those trees is parsed fresh each call,
since ``tmp_path`` files may change between reads.
"""

from __future__ import annotations

import ast
import functools
import gc
from collections.abc import Iterator
from pathlib import Path

import strands_robots

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent
_TESTS_ROOT = Path(__file__).resolve().parent
_SHARED_ROOTS = (_PACKAGE_ROOT, _TESTS_ROOT, _TESTS_ROOT.parent / "tests_integ")

#: Every shared tree by ``id()``. The trees live for the rest of the process,
#: so an id names one tree for as long as this dict can be asked.
_SHARED_TREES: dict[int, ast.Module] = {}
_WALKS: dict[int, tuple[ast.AST, ...]] = {}


@functools.cache
def _sources_under(root: Path) -> frozenset[str]:
    return frozenset(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))


@functools.cache
def _trees_under(root: Path) -> dict[str, ast.Module]:
    """Every file under ``root`` parsed once, keyed by its text, and kept out of the collector.

    The trees live for the rest of the process, so a cyclic collection would
    walk all of them again each time it runs: holding the 2,200-file test tree
    without this cost about as much collector time as the parses it saved.
    ``gc.freeze()`` after one collection moves them, and everything else alive
    at that point, to the permanent generation the collector never scans.
    """
    trees = {source: ast.parse(source) for source in _sources_under(root)}
    _SHARED_TREES.update((id(tree), tree) for tree in trees.values())
    gc.collect()
    gc.freeze()
    return trees


def _shared_tree(source: str, root: Path) -> ast.Module:
    tree = _trees_under(root).get(source)
    return tree if tree is not None else ast.parse(source)


def parse_file(path: Path) -> ast.Module:
    """Return the module tree of the Python source file at ``path``.

    Args:
        path: A ``.py`` file. Under the installed package, ``tests/`` or
            ``tests_integ/`` the tree is the one parsed for the whole tree
            the first time any of its files was asked for (a file edited
            since is parsed afresh); anywhere else it is parsed on every call.

    Returns:
        The parsed module. Do not mutate it; deep-copy first.
    """
    resolved = Path(path).resolve()
    source = resolved.read_text(encoding="utf-8")
    for root in _SHARED_ROOTS:
        if resolved.is_relative_to(root):
            return _shared_tree(source, root)
    return ast.parse(source)


def parse_source(source: str) -> ast.Module:
    """Return the module tree of ``source``, shared when it is a graded file's text.

    For graders whose helpers take the text rather than the path, so a planted
    negative can hand them a string: the text of a file under the package,
    ``tests/`` or ``tests_integ/`` is parsed once per process, any other text
    on every call.

    Args:
        source: Python source text.

    Returns:
        The parsed module. Do not mutate it; deep-copy first.
    """
    for root in _SHARED_ROOTS:
        if source in _sources_under(root):
            return _shared_tree(source, root)
    return ast.parse(source)


def walk_tree(node: ast.AST) -> Iterator[ast.AST]:
    """Yield what :func:`ast.walk` yields for ``node``, walking a shared tree once.

    More than 200 graders walk the same shared module trees, and
    :func:`ast.walk` is a Python generator that costs about as much as the
    parse it walks: the test tree alone is ~3 million nodes. The nodes of a
    tree :func:`parse_file` or :func:`parse_source` handed out are listed on
    the first walk and replayed, in the same breadth-first order, on every
    later one. Any other node - a
    subtree, a ``copy.deepcopy``, a planted-negative string's tree - is walked
    afresh, so a grader that edits its copy sees the edit.

    Args:
        node: The tree or node to walk.

    Returns:
        An iterator over ``node`` and every node below it.
    """
    if _SHARED_TREES.get(id(node)) is not node:
        return ast.walk(node)
    nodes = _WALKS.get(id(node))
    if nodes is None:
        nodes = _WALKS[id(node)] = tuple(ast.walk(node))
    return iter(nodes)
