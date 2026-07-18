"""Every mesh module - top-level *and* subpackage - must cross-reference sibling
code by module, never by source filename.

Citing a sibling source file (``core.py``, ``security.py``, ``session.py``, ...)
in a docstring is documentation archaeology: the name breaks silently the moment
a file is renamed or split, and it points a reader at a path instead of an
importable symbol. This package already carried a *dead* example - an
``_ProcessAuditState`` docstring pointed at ``mesh/security.py::_ProcessSecurityState``,
a class that no longer exists - which is exactly the silent-breakage this rule
prevents. The project convention (already guarded for the policy package by
``tests/policies/test_docstring_module_xrefs.py``, the trainer package by
``tests/training/test_docstring_module_xrefs.py``, and the MuJoCo backend by
``tests/simulation/mujoco/test_docstring_module_xrefs.py``) is to use Sphinx
cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` - that name the actual
API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring across the whole
``strands_robots.mesh`` package tree - the top-level modules (``core.py``,
``security.py``, ``audit.py``, ...) *and* the ``transport`` and ``iot``
subpackages - and fails if any embeds a ``<name>.py`` token that names an actual
module anywhere in that tree. An earlier version scanned only the top-level
modules, so a ``session.py`` filename reference in
``mesh/transport/zenoh_transport.py`` slipped through unguarded; walking the full
tree closes that gap. The check is intentionally sibling-aware rather than
flagging every ``.py`` token: mesh docstrings legitimately cite *test* modules by
filename (the guarding ``test_*.py`` files that pin a behavior), which live under
``tests/`` and are not importable modules of the package. ``__init__.py`` is
excluded because it names a package, not a cross-referenceable module symbol.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.mesh as mesh_pkg

# A bare source-filename token such as ``core.py`` or ``session.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(mesh_pkg.__file__).parent
# Every module across the mesh tree (top-level + transport/ + iot/ subpackages),
# minus ``__init__.py`` which names a package rather than a referenceable module.
_PACKAGE_MODULES = {p.name for p in _PACKAGE_DIR.rglob("*.py") if p.name != "__init__.py"}


def _docstrings_with_sibling_filename_refs() -> dict[str, list[str]]:
    """Map ``relpath::qualname`` -> module-filename tokens found in its docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            hits = [tok for tok in _FILENAME_RE.findall(doc) if tok in _PACKAGE_MODULES]
            if hits:
                qualname = getattr(node, "name", "<module>")
                rel = source_file.relative_to(_PACKAGE_DIR)
                offenders[f"{rel}::{qualname}"] = hits
    return offenders


def test_mesh_package_tree_scanned() -> None:
    """Guard: the scan walked the top-level modules and both subpackages."""
    assert {"core.py", "security.py", "audit.py"} <= _PACKAGE_MODULES
    assert {"zenoh_transport.py", "bridge_transport.py"} <= _PACKAGE_MODULES  # transport/
    assert {"provision.py", "bootstrap.py"} <= _PACKAGE_MODULES  # iot/


def test_mesh_docstrings_reference_modules_not_sibling_filenames() -> None:
    offenders = _docstrings_with_sibling_filename_refs()
    assert not offenders, (
        "mesh docstrings must cross-reference sibling modules by module "
        "(:mod:`~strands_robots.mesh.session`) not source filename (``session.py``). "
        f"Offending docstrings: {offenders}"
    )
