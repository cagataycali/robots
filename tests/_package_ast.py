"""The parsed tree of a package source file, shared by every grader in a process.

About 150 test modules grade the whole ``strands_robots`` tree by reading and
parsing every one of its ~380 files. Each did that on its own, many at module
scope, which pytest-xdist runs once per worker during collection, so a run
parsed the package several hundred times - most of the time those graders
spent. :func:`parse_file` parses a package file once per process and hands
every later caller the same tree; :func:`parse_source` does the same for a
grader that is handed the file's text instead of its path.

The tree is shared, so it is read-only by contract: a grader that edits one
(a planted negative, an ``ast.NodeTransformer``) works on
``copy.deepcopy(tree)``. A path outside the package is parsed fresh each call,
since test fixtures and ``tmp_path`` files may change between reads.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import strands_robots

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent


@functools.cache
def _parse_package_source(source: str) -> ast.Module:
    return ast.parse(source)


@functools.cache
def _package_sources() -> frozenset[str]:
    return frozenset(path.read_text(encoding="utf-8") for path in _PACKAGE_ROOT.rglob("*.py"))


def parse_file(path: Path) -> ast.Module:
    """Return the module tree of the Python source file at ``path``.

    Args:
        path: A ``.py`` file. Under the installed package its text is parsed
            once per process (an edited file is new text, so it is parsed
            again); anywhere else it is parsed on every call.

    Returns:
        The parsed module. Do not mutate it; deep-copy first.
    """
    resolved = Path(path).resolve()
    source = resolved.read_text(encoding="utf-8")
    if not resolved.is_relative_to(_PACKAGE_ROOT):
        return ast.parse(source)
    return _parse_package_source(source)


def parse_source(source: str) -> ast.Module:
    """Return the module tree of ``source``, shared when it is a package file's text.

    For graders whose helpers take the text rather than the path, so a planted
    negative can hand them a string: the text of a package file is parsed once
    per process, any other text on every call.

    Args:
        source: Python source text.

    Returns:
        The parsed module. Do not mutate it; deep-copy first.
    """
    if source in _package_sources():
        return _parse_package_source(source)
    return ast.parse(source)
