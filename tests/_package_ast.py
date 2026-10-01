"""The parsed tree of a package source file, shared by every grader in a process.

About 150 test modules grade the whole ``strands_robots`` tree by reading and
parsing every one of its ~380 files. Each did that on its own, many at module
scope, which pytest-xdist runs once per worker during collection, so a run
parsed the package several hundred times - most of the time those graders
spent. :func:`parse_file` parses a package file once per process and hands
every later caller the same tree.

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
def _parse_package_file(path: Path, mtime_ns: int, size: int) -> ast.Module:
    del mtime_ns, size  # part of the cache key only: an edited file is parsed again
    return ast.parse(path.read_text(encoding="utf-8"))


def parse_file(path: Path) -> ast.Module:
    """Return the module tree of the Python source file at ``path``.

    Args:
        path: A ``.py`` file. Under the installed package it is parsed once
            per process (keyed on its modification time and size); anywhere
            else it is parsed on every call.

    Returns:
        The parsed module. Do not mutate it; deep-copy first.
    """
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(_PACKAGE_ROOT):
        return ast.parse(resolved.read_text(encoding="utf-8"))
    stat = resolved.stat()
    return _parse_package_file(resolved, stat.st_mtime_ns, stat.st_size)
