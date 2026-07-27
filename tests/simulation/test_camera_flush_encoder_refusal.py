# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A refused encode is reported by the camera flush, never raised past it.

:func:`strands_robots.rendering.encode_clip` raises ``RuntimeError`` when the
encoder accepted the frames but wrote no clip (for example a frame size libx264
refuses), because the returned path is the caller's only handle on the
artifact. ``stop_cameras_recording`` is an agent-facing tool whose flush
contract is best-effort and never-raise, so every backend that flushes buffered
camera frames through ``encode_clip`` has to turn that refusal into a reported
artifact error rather than letting it escape the tool envelope.

The MuJoCo backend already wrapped the call in a best-effort handler; the Isaac
backend handled only ``ImportError``, so a refused encode would have propagated
out of the tool *and* the artifact line still claimed every buffered frame as
written. Checked by AST because the Isaac backend cannot be imported without the
Omniverse runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.simulation as simulation_pkg

_PACKAGE_DIR = Path(simulation_pkg.__file__).parent

# Backend module -> the modules that flush buffered camera frames to disk.
_FLUSH_MODULES = ["mujoco/rendering.py", "isaac/simulation.py"]


def _functions_calling_encode_clip(tree: ast.AST) -> list[ast.FunctionDef]:
    """Every function definition whose body calls ``encode_clip``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "encode_clip":
                found.append(node)
                break
    return found


def _catches_runtime_error(func: ast.FunctionDef) -> bool:
    """True if some ``try`` in the function handles ``RuntimeError``."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            names = []
            if isinstance(handler.type, ast.Name):
                names = [handler.type.id]
            elif isinstance(handler.type, ast.Tuple):
                names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
            if {"RuntimeError", "BaseException", "Exception"} & set(names):
                return True
    return False


def test_every_camera_flush_handles_an_encoder_refusal() -> None:
    """No backend lets encode_clip's refusal escape stop_cameras_recording."""
    checked = 0
    for relative in _FLUSH_MODULES:
        module_path = _PACKAGE_DIR / relative
        assert module_path.exists(), f"{relative} moved; update this guard"
        functions = _functions_calling_encode_clip(ast.parse(module_path.read_text(encoding="utf-8")))
        assert functions, f"{relative} no longer flushes through encode_clip; update this guard"
        for func in functions:
            assert _catches_runtime_error(func), (
                f"{relative}::{func.name} calls encode_clip without handling a RuntimeError, "
                "so a refused encode would raise past the tool envelope"
            )
            checked += 1
    assert checked >= len(_FLUSH_MODULES)
