# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark-adapter public API must document every public member.

The :mod:`strands_robots.benchmarks` package is the library home of the
per-benchmark adapters layered on :mod:`strands_robots.simulation.benchmark`.
The LIBERO adapter is its first citizen: :class:`~strands_robots.benchmarks.libero.adapter.LiberoAdapter`
(a ``BenchmarkProtocol`` built around a BDDL task), the BDDL parser AST nodes
(:class:`~strands_robots.benchmarks.libero.bddl_parser.Pred` and the
:class:`~strands_robots.benchmarks.libero.bddl_parser.And` /
:class:`~strands_robots.benchmarks.libero.bddl_parser.Or` /
:class:`~strands_robots.benchmarks.libero.bddl_parser.Not` combinators) and the
:func:`~strands_robots.benchmarks.libero.suite.load_libero_suite` loader.
Agents and integrators read these docstrings to drive the surface, so each
public class, method/property, and module-level function needs its own
docstring rather than silently leaning on inherited protocol text -- a
``supported_robots`` override, for instance, should state its own robot-set
contract rather than echoing the abstract property.

This guard walks the package modules by AST (no import, so it never needs the
optional ``benchmark-libero`` extra installed) and fails if any public class,
public method/property, or public module-level function lacks a docstring.
Nested closures (functions defined inside a method body) are out of scope:
they are implementation detail, not public API.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.benchmarks as benchmarks_pkg

_PACKAGE_DIR = Path(benchmarks_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional backend installed.
_MODULES = (
    "libero/adapter.py",
    "libero/bddl_parser.py",
    "libero/suite.py",
)

# Every public class the package exposes, keyed ``module.py::ClassName``. Pinned
# so a refactor that drops or renames a class trips the completeness guard
# instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "libero/adapter.py::LiberoAdapter",
    "libero/bddl_parser.py::BDDLParseError",
    "libero/bddl_parser.py::Pred",
    "libero/bddl_parser.py::And",
    "libero/bddl_parser.py::Or",
    "libero/bddl_parser.py::Not",
    "libero/bddl_parser.py::BDDLProblem",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "libero/adapter.py::camera_config_error",
    "libero/bddl_parser.py::parse_bddl",
    "libero/bddl_parser.py::parse_bddl_file",
    "libero/bddl_parser.py::compile_goal",
    "libero/suite.py::load_libero_suite",
    "libero/suite.py::available_suites",
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


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Map public top-level class name -> node for one module tree."""
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef) and not node.name.startswith("_")}


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map public top-level function name -> node for one module tree."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
    }


def test_public_classes_are_the_pinned_set() -> None:
    """The scanned public classes must match the pinned expectation exactly.

    A drift in either direction (a new undocumented class, or a rename that
    orphans an entry) fails here so the docstring guard can never silently
    stop covering a class.
    """
    found = {f"{module}::{name}" for module in _MODULES for name in _classes(_module_tree(module))}
    assert found == _EXPECTED_CLASSES, f"public class set drifted: {found ^ _EXPECTED_CLASSES}"


def test_public_functions_are_the_pinned_set() -> None:
    """The scanned public module-level functions must match the pinned expectation."""
    found = {f"{module}::{name}" for module in _MODULES for name in _functions(_module_tree(module))}
    assert found == _EXPECTED_FUNCTIONS, f"public function set drifted: {found ^ _EXPECTED_FUNCTIONS}"


def test_public_classes_have_docstrings() -> None:
    """Every public class carries a class docstring."""
    offenders = [
        f"{module}::{name}"
        for module in _MODULES
        for name, node in _classes(_module_tree(module)).items()
        if ast.get_docstring(node) is None
    ]
    assert not offenders, f"public classes missing docstrings: {offenders}"


def test_public_methods_and_properties_have_docstrings() -> None:
    """Every public method/property of every public class carries a docstring."""
    offenders: list[str] = []
    for module in _MODULES:
        for cls_name, node in _classes(_module_tree(module)).items():
            offenders += [f"{module}::{cls_name}.{m}" for m in _public_members_without_docstring(node)]
    assert not offenders, f"public methods/properties missing docstrings: {offenders}"


def test_public_functions_have_docstrings() -> None:
    """Every public module-level function carries a docstring."""
    offenders = [
        f"{module}::{name}"
        for module in _MODULES
        for name, node in _functions(_module_tree(module)).items()
        if ast.get_docstring(node) is None
    ]
    assert not offenders, f"public functions missing docstrings: {offenders}"
