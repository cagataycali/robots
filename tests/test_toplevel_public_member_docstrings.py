# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public member of the top-level ``strands_robots`` modules must have a docstring.

The top-level modules are the surface an agent reaches first:
:func:`~strands_robots.robot.Robot` and the hardware
:class:`~strands_robots.hardware_robot.Robot`, the recording/streaming readers
(:class:`~strands_robots.dataset_recorder.DatasetRecorder`,
:class:`~strands_robots.streaming_dataset.StreamingDatasetReader`), and the
``python -m strands_robots`` CLI entry points (:mod:`strands_robots.__main__`,
:func:`strands_robots.doctor.main`). Several of these carried a rich class or
module docstring but left individual accessors (``repo_id`` / ``root`` /
``num_frames`` / ``num_episodes`` / ``fps`` / ``tool_name`` / ``tool_type``) and
the CLI ``main`` entry points undocumented, so a reader driving the surface
blind could not tell what an accessor returns without reading its body.

This mirrors the device_connect guard
(``tests/test_device_connect_public_member_docstrings.py``), the mesh guard
(``tests/mesh/test_public_member_docstrings.py``), and the policy guard
(``tests/policies/test_builtin_policy_docstrings.py``), extending the
completeness bar to the flat top-level modules.

The check walks the top-level modules by AST (no import, so it needs none of
the optional extras installed) and fails if any public module-level function or
any public method/property of a public class defines no docstring. Private
modules (``_async_utils`` / ``_dyld``) and the re-export-only ``__init__`` are
out of scope; ``__main__`` is included because it is the ``python -m`` entry
point. ``@overload`` stubs (the :func:`~strands_robots.robot.Robot` factory
overloads document their contract once on the implementation) and property
setters/deleters (a ``@x.setter`` mirrors its documented getter) are exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots

_PACKAGE_DIR = Path(strands_robots.__file__).parent

# Anchors that must always be found by the walk, so a refactor that renames or
# relocates the surface trips this guard instead of silently scanning nothing.
_EXPECTED_CLASSES = {
    "dataset_recorder.py::DatasetRecorder",
    "hardware_robot.py::Robot",
    "streaming_dataset.py::StreamingDatasetReader",
}
_EXPECTED_FUNCTIONS = {
    "__main__.py::main",
    "doctor.py::main",
    "robot.py::Robot",
    "verify_dataset.py::verify_dataset",
}


def _toplevel_modules() -> list[Path]:
    """Return the top-level ``strands_robots`` module files in scope.

    Subpackages have their own guards; private ``_*`` modules and the
    re-export-only ``__init__`` are excluded, but the ``__main__`` entry point
    is included.
    """
    modules: list[Path] = []
    for path in sorted(_PACKAGE_DIR.glob("*.py")):
        name = path.name
        if name == "__init__.py":
            continue
        if name.startswith("_") and name != "__main__.py":
            continue
        modules.append(path)
    return modules


def _is_setter_or_deleter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@<name>.setter`` or ``@<name>.deleter``."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
            return True
    return False


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@typing.overload`` stub (documented once on the impl)."""
    for dec in node.decorator_list:
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
        if name == "overload":
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
        if _is_setter_or_deleter(node) or _is_overload(node):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every public top-level class."""
    classes: dict[str, ast.ClassDef] = {}
    for path in _toplevel_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{path.name}::{node.name}"] = node
    return classes


def _public_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``module.py::func`` -> FunctionDef for every public module-level function.

    A ``@overload``-decorated stub is keyed by name like any other; multiple
    overloads collapse onto the single key, and the check ignores them anyway.
    """
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for path in _toplevel_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                funcs[f"{path.name}::{node.name}"] = node
    return funcs


def test_toplevel_modules_resolve() -> None:
    """The top-level module set resolves and contains the pinned anchors."""
    names = {p.name for p in _toplevel_modules()}
    assert "__main__.py" in names, "__main__ entry point not scanned"
    classes = _public_classes()
    funcs = _public_functions()
    assert _EXPECTED_CLASSES <= set(classes), f"missing expected classes: {_EXPECTED_CLASSES - set(classes)}"
    assert _EXPECTED_FUNCTIONS <= set(funcs), f"missing expected functions: {_EXPECTED_FUNCTIONS - set(funcs)}"


def test_every_public_class_member_has_docstring() -> None:
    """No public method/property of a public top-level class may lack a docstring."""
    missing: dict[str, list[str]] = {}
    for qualname, class_node in _public_classes().items():
        offenders = _public_members_without_docstring(class_node)
        if offenders:
            missing[qualname] = offenders

    assert not missing, "Public class members missing docstrings:\n" + "\n".join(
        f"  {qualname}: {', '.join(names)}" for qualname, names in sorted(missing.items())
    )


def test_every_public_module_function_has_docstring() -> None:
    """No public module-level function of a top-level module may lack a docstring."""
    missing: list[str] = []
    for qualname, node in _public_functions().items():
        if _is_overload(node):
            continue
        if ast.get_docstring(node) is None:
            missing.append(qualname)

    assert not missing, "Public module functions missing docstrings:\n" + "\n".join(f"  {q}" for q in sorted(missing))
