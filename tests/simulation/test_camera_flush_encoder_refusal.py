# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A refused encode is reported by the camera flush, never raised past it.

:func:`strands_robots.rendering.encode_clip` refuses to hand back a clip it
cannot stand behind - ``RuntimeError`` when the encoder accepted the frames but
wrote no file (for example a frame size libx264 refuses), ``ValueError`` when it
will not encode at the requested rate - because the returned path is the
caller's only handle on the artifact. ``stop_cameras_recording`` is an
agent-facing tool whose flush contract is best-effort and never-raise, so every
backend that flushes buffered camera frames through ``encode_clip`` has to turn
*both* refusals into a reported artifact error rather than letting one escape
the tool envelope - the more so because the flush runs after the recording state
is cleared, so an escape loses every buffered frame with no structured response.

This is a cross-backend structural guard: it asserts the property holds for
every module that flushes, so a new backend cannot add an unguarded flush. The
per-backend behavioural coverage lives next to each backend
(``tests/simulation/isaac/test_cameras_recording_preflight_guards.py``,
``tests/simulation/mujoco/test_recording_backends.py``); this one is by AST so
it stays honest about backends whose flush path a unit test cannot reach.
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


# Every exception ``encode_clip`` raises to refuse a clip, and the broader
# handlers that also contain them.
_REFUSALS = ("RuntimeError", "ValueError")
_CATCH_ALL = {"BaseException", "Exception"}


def _handled_refusals(func: ast.FunctionDef) -> set[str]:
    """The ``encode_clip`` refusals some ``try`` in the function handles."""
    handled: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            names = []
            if isinstance(handler.type, ast.Name):
                names = [handler.type.id]
            elif isinstance(handler.type, ast.Tuple):
                names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
            if _CATCH_ALL & set(names):
                return set(_REFUSALS)
            handled |= set(names) & set(_REFUSALS)
    return handled


def test_every_camera_flush_handles_an_encoder_refusal() -> None:
    """No backend lets either encode_clip refusal escape stop_cameras_recording."""
    checked = 0
    for relative in _FLUSH_MODULES:
        module_path = _PACKAGE_DIR / relative
        assert module_path.exists(), f"{relative} moved; update this guard"
        functions = _functions_calling_encode_clip(ast.parse(module_path.read_text(encoding="utf-8")))
        assert functions, f"{relative} no longer flushes through encode_clip; update this guard"
        for func in functions:
            missing = set(_REFUSALS) - _handled_refusals(func)
            assert not missing, (
                f"{relative}::{func.name} calls encode_clip without handling {sorted(missing)}, "
                "so a refused encode would raise past the tool envelope"
            )
            checked += 1
    assert checked >= len(_FLUSH_MODULES)
