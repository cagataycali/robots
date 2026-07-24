# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public rendering class must document its public methods/properties.

The hybrid-render layer (``strands_robots.rendering``) is a public API surface:
:class:`~strands_robots.rendering.CameraParams`, the
:class:`~strands_robots.rendering.BackgroundRenderer` protocol with its
:class:`~strands_robots.rendering.PanoramaBackground` (zero-ML) and
:class:`~strands_robots.rendering.GsplatBackground` (3DGS) implementations, and
the :class:`~strands_robots.rendering.HybridCompositor` depth-compositor.
Implementers pick these members off the concrete class, so an override with no
docstring (e.g. a ``render`` that silently leans on the protocol's inherited
one) leaves the concrete return/exception contract undocumented at the call
site. ``PanoramaBackground.render`` returns depth pinned at ``cam.zfar`` while
``GsplatBackground.render`` returns metric depth and can raise on a CUDA/file
error -- behaviour worth stating per implementation, not just on the protocol.

This guard walks the rendering modules by AST (no import, so it needs neither
the ``sim-gs`` extra nor a GPU) and fails if any public method or property of a
public rendering class defines no docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.rendering as rendering_pkg

_PACKAGE_DIR = Path(rendering_pkg.__file__).parent

# Modules that define the package's public classes. media utils (video.py) are
# free functions, guarded elsewhere; this guard is about class member coverage.
_MODULES = ("backgrounds.py", "camera.py", "compositor.py")

# The public classes this guard must actually find -- pins the guard against
# silently going vacuous if a class is renamed/moved out of these modules.
_EXPECTED_CLASSES = {
    "backgrounds.py::BackgroundRenderer",
    "backgrounds.py::PanoramaBackground",
    "backgrounds.py::GsplatBackground",
    "camera.py::CameraParams",
    "compositor.py::FrameSource",
    "compositor.py::CompositeFrame",
    "compositor.py::HybridCompositor",
}


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring."""
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
        source_file = _PACKAGE_DIR / module
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def test_expected_public_classes_are_present() -> None:
    """The guard must actually see every documented public class (not vacuous)."""
    found = set(_public_classes())
    missing = _EXPECTED_CLASSES - found
    assert not missing, f"expected public rendering classes not found by AST walk: {sorted(missing)}"


def test_every_public_rendering_member_has_a_docstring() -> None:
    """No public method/property of a public rendering class may be undocumented.

    Fails pre-fix on ``PanoramaBackground.render`` / ``GsplatBackground.render``,
    which shipped without docstrings despite being the concrete public render
    entrypoints of the ``BackgroundRenderer`` contract.
    """
    offenders = {
        name: missing
        for name, node in _public_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, f"public rendering members missing docstrings: {offenders}"
