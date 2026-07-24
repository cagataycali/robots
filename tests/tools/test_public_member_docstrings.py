"""Every public class in the ``strands_robots.tools`` package must document each
of its public methods and properties.

The tools package is the agent-facing command surface: each ``@tool`` entry
point plus the small stateful helpers those tools drive. One such helper,
:class:`~strands_robots.tools.lerobot_train.SessionManager`, tracks detached
training runs on disk (add / get / list / remove) but left its public lifecycle
methods undocumented, so a reader driving the training-session API blind could
not tell what ``get_session`` returns for an unknown name, or that
``remove_session`` is a no-op when the name is untracked, without reading the
body. This mirrors the mesh-package guard
(``tests/mesh/test_public_member_docstrings.py``), the policy-package guard
(``tests/policies/test_builtin_policy_docstrings.py``), and the tools docstring
cross-reference guard (``tests/tools/test_docstring_module_xrefs.py``).

The check walks every module across the tools tree by AST (no import, so it
needs none of the optional tool backends installed) and fails if any public
method or property of a public class defines no docstring. Property setters and
deleters are exempt: a ``@x.setter`` mirrors its documented getter and a
docstring on it is noise, matching the convention elsewhere in the tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.tools as tools_pkg

_PACKAGE_DIR = Path(tools_pkg.__file__).parent


def _is_setter_or_deleter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@<name>.setter`` or ``@<name>.deleter``."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
            return True
    return False


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring."""
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if _is_setter_or_deleter(node):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``relpath::ClassName`` -> ClassDef for every public class across the tools tree."""
    classes: dict[str, ast.ClassDef] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        rel = source_file.relative_to(_PACKAGE_DIR)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{rel}::{node.name}"] = node
    return classes


def test_tools_tree_has_public_classes() -> None:
    """Guard: the scan actually walked the tools tree and found the class to protect."""
    found = set(_public_classes())
    assert any(q.endswith("::SessionManager") for q in found), found


def test_tools_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _public_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a public tools class must have a "
        "docstring describing what it returns or does (a rich class docstring "
        "does not document its individual accessors). Undocumented members: " + repr(offenders)
    )
