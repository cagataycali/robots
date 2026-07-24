"""Every public class in the ``strands_robots.mesh`` package must document each
of its public methods and properties.

The mesh package is the fleet-coordination surface an agent drives directly -
:class:`~strands_robots.mesh.core.Mesh` (presence, pub/sub, peer discovery), the
teleop input pump (:class:`~strands_robots.mesh.input.InputPublisher` /
:class:`~strands_robots.mesh.input.InputReceiver`), and the pluggable transports
(:class:`~strands_robots.mesh.transport.base.MeshTransport` and its Zenoh / AWS
IoT / bridge implementations). Several classes carried a rich class docstring
but left public accessors - ``Mesh.alive`` / ``Mesh.peers``, the publisher /
receiver ``topic`` / ``stats`` properties, ``SubHandle.undeclare``, the
``BridgeTransport`` inspection properties, ``IotMqttTransport.thing_name`` -
undocumented, so a reader driving the API blind could not tell what an accessor
returns without reading its body. This mirrors the policy-package guard
(``tests/policies/test_builtin_policy_docstrings.py``) and the mesh docstring
cross-reference guard (``tests/mesh/test_docstring_module_xrefs.py``).

The check walks every module across the mesh tree by AST (no import, so it needs
none of the optional mesh backends - zenoh, awscrt - installed) and fails if any
public method or property of a public class defines no docstring. Property
setters and deleters are exempt: a ``@x.setter`` mirrors its documented getter
and a docstring on it is noise, matching the convention elsewhere in the tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.mesh as mesh_pkg

_PACKAGE_DIR = Path(mesh_pkg.__file__).parent


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
    """Map ``relpath::ClassName`` -> ClassDef for every public class across the mesh tree."""
    classes: dict[str, ast.ClassDef] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        rel = source_file.relative_to(_PACKAGE_DIR)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{rel}::{node.name}"] = node
    return classes


def test_mesh_tree_has_public_classes() -> None:
    """Guard: the scan actually walked the mesh tree and found public classes to protect."""
    found = set(_public_classes())
    assert any(q.endswith("::Mesh") for q in found), found
    assert any(q.endswith("::InputPublisher") for q in found), found
    assert any(q.endswith("::BridgeTransport") for q in found), found


def test_mesh_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _public_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a public mesh class must have a "
        "docstring describing what it returns or does (a rich class docstring "
        "does not document its individual accessors). Undocumented members: " + repr(offenders)
    )
