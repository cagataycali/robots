# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The asset-resolution public API must document every public member.

The ``strands_robots.assets`` package is how a robot's MuJoCo model files are
found on disk: :mod:`~strands_robots.assets.manager` resolves a robot name to a
cached model path (``robot_descriptions`` package or the local assets dir), and
:mod:`~strands_robots.assets.download` fetches missing meshes/XML from MuJoCo
Menagerie (and other declared sources) into that cache. First-time users and
agents call these functions directly to discover, resolve, and pre-download the
assets a sim needs, so each public entry point must state its OWN behavior
rather than leave a caller guessing what a name resolves to or where a download
lands.

This guard walks the package modules by AST (no import of the module bodies, so
it never needs an optional dependency installed) and fails if any public class,
public method/property, or public module-level function lacks a docstring. It
also pins the discovered public surface so a refactor that drops or renames an
entry point trips the completeness guard instead of silently shrinking the scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.assets as assets_pkg

_PACKAGE_DIR = Path(assets_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional dependency installed.
_MODULES = ("manager.py", "download.py")

# Every public class the package exposes, keyed ``module.py::ClassName``. The
# package exposes only module-level functions today, so this is empty; a future
# public class trips this pinned-empty set and gets its own coverage here.
_EXPECTED_CLASSES: set[str] = set()

# Every public module-level function the package exposes. Pinned so a refactor
# that drops or renames a function trips the completeness guard instead of
# silently shrinking the scan.
_EXPECTED_FUNCTIONS = {
    "manager.py::is_robot_asset_present",
    "manager.py::resolve_model_path",
    "manager.py::resolve_model_dir",
    "manager.py::get_robot_info",
    "manager.py::list_available_robots",
    "download.py::auto_download_robot",
    "download.py::download_robots",
}


def _module_tree(module: str) -> ast.Module:
    """Parse one package module into an AST (no import)."""
    source_file = _PACKAGE_DIR / module
    return ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder methods (``__init__`` and friends) are out of scope: their contract
    is documented on the class docstring itself.
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
    """Map ``module.py::ClassName`` -> ClassDef for every public class in the modules."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def _public_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``module.py::func`` -> FunctionDef for every public module-level function."""
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                funcs[f"{module}::{node.name}"] = node
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
        "Every public class in strands_robots.assets -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior. Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.assets must have "
        "a docstring describing what it resolves/downloads and where. "
        "Undocumented functions: " + repr(offenders)
    )
