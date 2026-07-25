# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The remote-inference public API must document every public member.

The ``strands_robots.inference`` package is the client/server split that lets
an edge robot host forward observations to a remote GPU running a large VLA:
:class:`~strands_robots.inference.server.PolicyServer` wraps any
:class:`~strands_robots.policies.base.Policy`, and
:class:`~strands_robots.inference.client.RemotePolicy` is the drop-in ``Policy``
that forwards over the WS-JSON protocol in
:mod:`~strands_robots.inference.protocol`. Agents and integrators read these
docstrings to drive the two-machine surface, so each concrete ``Policy``
override on ``RemotePolicy`` must state its OWN behavior (it forwards to the
server) rather than silently leaning on the base ABC's local-only text.

This guard walks the package modules by AST (no import, so it never needs the
optional ``inference`` extra installed) and fails if any public class, public
method/property, or public module-level function lacks a docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.inference as inference_pkg

_PACKAGE_DIR = Path(inference_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional dependency installed.
_MODULES = ("client.py", "server.py", "protocol.py")

# Every public class the package exposes, keyed ``module.py::ClassName``. Pinned
# so a refactor that drops or renames a class trips the completeness guard
# instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "client.py::RemotePolicy",
    "server.py::PolicyServer",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "server.py::main",
    "protocol.py::encode_ndarray",
    "protocol.py::decode_ndarray",
    "protocol.py::dumps",
    "protocol.py::loads",
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
        "Every public class in strands_robots.inference -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior (concrete Policy overrides on RemotePolicy must not lean on "
        "the base ABC's local-only text). Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.inference must "
        "have a docstring. Undocumented functions: " + repr(offenders)
    )
