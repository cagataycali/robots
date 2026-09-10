"""Every module in the package cites internal code by module, never by source filename.

Citing an internal source file (``core.py``, ``drivers/base.py``) in a docstring or
a comment is documentation archaeology: the name breaks silently the moment a file
is renamed or split, and it points a reader at a path instead of an importable
symbol. The convention is a Sphinx cross-reference role - ``:mod:``, ``:class:``,
``:meth:``, ``:func:`` - that names the actual API object, so the reference is
checkable and survives refactors. A citation of a file in *another* project stays a
path, because no importable strands symbol addresses it; qualifying it with its
project (``lerobot/policies/factory.py``) is what makes it unambiguous.

One rule, one grader, the whole tree. This replaced nine near-identical per-package
copies whose scopes had drifted apart: four flagged every ``.py`` token, three only
sibling filenames, one every module in its subtree, one every stem in the package;
one scanned comments as well as docstrings, eight did not. The drift was accidental
- each copy was the previous one retuned - and it cost accuracy in both directions.
Too strict: flagging every ``.py`` token cannot tell an internal sibling from an
upstream script, so a docstring citing GR00T's ``examples/Libero/eval/utils.py``
was an offender only because the package happens to ship a ``utils.py``. Too
narrow: nine files guarded nine packages, so the same rot went unread in
``dashboard/``, ``device_connect/``, ``drivers/``, ``rendering/`` and the top-level
modules, and comment references were unguarded everywhere but ``mesh/``.

Resolution is by path, not by name: a citation is internal when the path it spells
resolves to a module that actually ships in the package, so an external path is
left alone even when its last component collides with an internal module name.
``__init__.py`` and ``__main__.py`` are never offenders - they name a package and
an entry point, not a module symbol a ``:mod:`` role can address.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

import strands_robots

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent
_PACKAGE_NAME = _PACKAGE_DIR.name

# An optional POSIX path prefix plus a module filename: ``drivers/base.py``, ``core.py``.
_CITATION_RE = re.compile(r"\b((?:[A-Za-z0-9_.-]+/)*)([A-Za-z_][A-Za-z0-9_]*\.py)\b")

# These name a package and an entry point rather than a module symbol, so no
# cross-reference role can replace them and they are never offenders.
_UNADDRESSABLE = frozenset({"__init__.py", "__main__.py"})

_SOURCE_FILES = sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
# Every module path relative to the package root, for resolving a qualified citation.
_RELPATHS = {p.relative_to(_PACKAGE_DIR).as_posix() for p in _SOURCE_FILES}
# Bare filenames of addressable modules, for resolving an unqualified citation.
_MODULE_FILENAMES = {p.name for p in _SOURCE_FILES if p.name not in _UNADDRESSABLE}

# The packages that each used to carry their own copy of this guard. Pinned so a
# future refactor cannot silently shrink the scope back to a subset of the tree.
_PREVIOUSLY_GUARDED = (
    "assets",
    "mesh",
    "policies",
    "registry",
    "simulation",
    "simulation/isaac",
    "simulation/mujoco",
    "tools",
    "training",
)


def _resolves_internally(cited: str) -> bool:
    """Whether a path-qualified citation names a module that ships in the package.

    Matching is on the path tail, so a citation relative to any directory in the
    tree resolves (``mujoco/rendering.py`` is
    ``simulation/mujoco/rendering.py``) while a same-named file in another
    project does not (``lerobot/policies/factory.py`` matches no tail).

    Args:
        cited: The cited path, e.g. ``drivers/base.py``.

    Returns:
        True when the citation names an internal module.
    """
    tail = cited.removeprefix(f"{_PACKAGE_NAME}/")
    return tail in _RELPATHS or any(rel.endswith(f"/{tail}") for rel in _RELPATHS)


def _internal_citations(text: str) -> list[str]:
    """Source-filename citations in ``text`` that resolve to a module in the package.

    Args:
        text: A docstring or comment body.

    Returns:
        The sorted, deduplicated cited paths that name an internal module.
    """
    found: set[str] = set()
    for prefix, filename in _CITATION_RE.findall(text):
        if filename in _UNADDRESSABLE:
            continue
        if not prefix:
            if filename in _MODULE_FILENAMES:
                found.add(filename)
        elif _resolves_internally(f"{prefix}{filename}"):
            found.add(f"{prefix}{filename}")
    return sorted(found)


def _docstring_offenders() -> dict[str, list[str]]:
    """Map ``relpath::qualname`` -> internal citations found in that docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in _SOURCE_FILES:
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            if citations := _internal_citations(doc):
                rel = source_file.relative_to(_PACKAGE_DIR).as_posix()
                offenders[f"{rel}::{getattr(node, 'name', '<module>')}"] = citations
    return offenders


def _comment_offenders() -> dict[str, list[str]]:
    """Map ``relpath`` -> internal citations found in that module's ``#`` comments.

    Uses :mod:`tokenize` so only real comment text is read: a filename used as
    data in a string literal names a file in another process, not an internal
    sibling, and is correct as written.
    """
    offenders: dict[str, list[str]] = {}
    for source_file in _SOURCE_FILES:
        source = source_file.read_text(encoding="utf-8")
        citations: set[str] = set()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                citations.update(_internal_citations(token.string))
        if citations:
            offenders[source_file.relative_to(_PACKAGE_DIR).as_posix()] = sorted(citations)
    return offenders


@pytest.mark.parametrize("package", _PREVIOUSLY_GUARDED)
def test_the_scan_reaches_every_package_that_once_guarded_itself(package: str) -> None:
    """The one grader covers every package that used to carry its own copy."""
    package_dir = _PACKAGE_DIR / package
    assert package_dir.is_dir(), f"{package} is no longer a package"
    modules = {p for p in package_dir.glob("*.py")}
    assert modules, f"{package} ships no modules"
    assert modules <= set(_SOURCE_FILES), f"{package} modules are outside the scan"


def test_the_scan_reaches_the_whole_tree_not_just_a_package() -> None:
    """Sanity on the scan basis: the whole tree, top-level modules included."""
    assert len(_SOURCE_FILES) > 200
    assert _PACKAGE_DIR / "robot.py" in set(_SOURCE_FILES)
    assert "mesh/core.py" in _RELPATHS
    assert "drivers/base.py" in _RELPATHS


@pytest.mark.parametrize(
    "citation",
    [
        "core.py",
        "a bare ``base.py`` reference",
        "mesh/core.py",
        "drivers/base.py",
        f"{_PACKAGE_NAME}/simulation/policy_runner.py",
        # Relative to a directory inside the tree, not to the package root.
        "mujoco/rendering.py",
    ],
)
def test_a_citation_that_resolves_to_an_internal_module_is_an_offender(citation: str) -> None:
    assert _internal_citations(citation)


@pytest.mark.parametrize(
    "citation",
    [
        # Upstream scripts whose last component collides with an internal module.
        "examples/Libero/eval/utils.py:get_libero_image()",
        "lerobot/policies/factory.py",
        "ProtoMotions deployment/motion_utils.py",
        "cagataycali/neon-the-g1/tools/use_unitree.py",
        # A package and an entry point, which no :mod: role addresses.
        "__init__.py",
        "strands_robots/simulation/__init__.py",
        "argv is ['/path/to/pkg/__main__.py', ...]",
        # Names no module in the package at all.
        "train.py",
        "inference_service.py",
    ],
)
def test_a_citation_that_names_no_internal_module_is_left_alone(citation: str) -> None:
    assert not _internal_citations(citation)


def test_docstrings_cite_modules_not_internal_filenames() -> None:
    offenders = _docstring_offenders()
    assert not offenders, (
        "Docstrings must cite internal code by module "
        "(:class:`~strands_robots.simulation.base.SimEngine`) not source filename "
        f"(``base.py``). Offending docstrings: {offenders}"
    )


def test_comments_cite_modules_not_internal_filenames() -> None:
    offenders = _comment_offenders()
    assert not offenders, (
        "Comments must cite internal code by module path "
        "(``strands_robots.mesh.audit._ensure_paths``) not source filename "
        f"(``audit.py``). Offending files: {offenders}"
    )
