# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pure-RTPS ROS 2 public API must document every public member.

``strands_robots.rtps`` lets an agent join a ROS 2 graph as a bare DDS
participant with no rclpy: :mod:`~strands_robots.rtps.mangling` maps ROS 2
graph names onto DDS topic/type names, and :mod:`strands_robots.rtps.idl`
ships the curated IDL message bundle an agent publishes and subscribes with.
Agents read these docstrings to drive the surface blind, so each public
mangling function, the bundle resolver functions, and every message dataclass
in the bundle needs its own docstring stating the ROS 2 type it mirrors.

This guard walks the package modules by AST (no import), so it never needs the
optional ``cyclonedds`` (``[ros2]`` extra) installed. The message dataclasses
are defined under a module-level ``if _HAVE_CYCLONEDDS:`` guard, so the scan
descends into that block as well as the module top level, and fails if any
pinned public class or function lacks a docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.rtps as rtps_pkg

_PACKAGE_DIR = Path(rtps_pkg.__file__).parent

# Public-API source files, keyed by the label used in the pinned sets below.
# ``__init__`` only re-exports the mangling helpers, so it defines no members.
_MODULES = {
    "mangling.py": _PACKAGE_DIR / "mangling.py",
    "idl": _PACKAGE_DIR / "idl" / "__init__.py",
}

# Every public message dataclass the IDL bundle exposes, plus the resolver and
# the mangling helpers. Pinned so a refactor that drops or renames one trips the
# completeness guard instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "idl::Vector3",
    "idl::Twist",
    "idl::Point",
    "idl::Quaternion",
    "idl::Pose",
    "idl::Time",
    "idl::Header",
    "idl::JointState",
    "idl::Image",
}

_EXPECTED_FUNCTIONS = {
    "mangling.py::dds_topic_name",
    "mangling.py::ros_topic_name",
    "mangling.py::dds_type_name",
    "idl::have_cyclonedds",
    "idl::get_type",
}


def _module_tree(path: Path) -> ast.Module:
    """Parse one package module into an AST (no import)."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _iter_defs(module: ast.Module) -> list[ast.stmt]:
    """Yield class/function defs at module top level and inside its top-level ``if`` blocks.

    The IDL message dataclasses live under ``if _HAVE_CYCLONEDDS:`` so that the
    module imports cleanly without the optional ``cyclonedds`` dependency; the
    walk descends one level into module-level ``if`` bodies to reach them.
    """
    defs: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            defs.append(node)
        elif isinstance(node, ast.If):
            for inner in [*node.body, *node.orelse]:
                if isinstance(inner, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    defs.append(inner)
    return defs


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder methods are out of scope: their contract is documented on the class
    docstring itself.
    """
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``label::ClassName`` -> ClassDef for every public class in the modules."""
    classes: dict[str, ast.ClassDef] = {}
    for label, path in _MODULES.items():
        for node in _iter_defs(_module_tree(path)):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{label}::{node.name}"] = node
    return classes


def _public_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``label::func`` -> FunctionDef for every public module-level function."""
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for label, path in _MODULES.items():
        for node in _iter_defs(_module_tree(path)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                funcs[f"{label}::{node.name}"] = node
    return funcs


def test_modules_define_expected_public_surface() -> None:
    """Guard: the scan actually found the classes and functions it protects."""
    assert set(_public_classes()) == _EXPECTED_CLASSES, set(_public_classes())
    assert set(_public_functions()) == _EXPECTED_FUNCTIONS, set(_public_functions())


def test_public_classes_and_members_have_docstrings() -> None:
    offenders: dict[str, list[str]] = {}
    for qualname, node in _public_classes().items():
        missing = _public_members_without_docstring(node)
        if ast.get_docstring(node) is None:
            missing = ["<class docstring>", *missing]
        if missing:
            offenders[qualname] = missing
    assert not offenders, (
        "Every public class in strands_robots.rtps -- including each IDL "
        "message dataclass -- must have a docstring naming the ROS 2 type it "
        "mirrors. Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.rtps must have a "
        "docstring. Undocumented functions: " + repr(offenders)
    )
