# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin that set_torque holds the engine lock while mutating model gain arrays.

The MuJoCo model's actuator gainprm/biasprm arrays are shared with the
engine's mj_step and render threads.  A write without the engine's _lock
races those threads and can leave half-applied gains visible mid-step,
silently corrupting the physics.  This test asserts the structural property:
set_torque acquires the engine lock before any gainprm/biasprm write.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path


def _source_path() -> Path:
    """Resolve the twin module's source file from the package tree."""
    twin_pkg = Path(__file__).resolve().parents[2] / "strands_robots" / "drivers" / "feetech" / "twin.py"
    assert twin_pkg.is_file(), f"twin.py not found at {twin_pkg}"
    return twin_pkg


def _method_ast(source: str, method_name: str) -> ast.FunctionDef | None:
    """Return the AST node of a method in the given source text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


class TestSetTorqueHoldsEngineLock:
    """set_torque must hold the engine's lock while mutating gainprm/biasprm."""

    def test_gainprm_write_is_inside_a_with_block(self) -> None:
        """The gainprm/biasprm assignments must be lexically inside a ``with`` statement."""
        source = _source_path().read_text()
        method = _method_ast(source, "set_torque")
        assert method is not None, "set_torque method not found in twin.py"

        # Walk the method body looking for attribute assignments to gainprm/biasprm.
        gain_writes: list[ast.Assign | ast.AugAssign] = []
        for node in ast.walk(method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                        if target.value.attr in ("gainprm", "biasprm"):
                            gain_writes.append(node)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Attribute):
                    if node.target.value.attr in ("gainprm", "biasprm"):
                        gain_writes.append(node)

        assert gain_writes, "Expected gainprm/biasprm writes in set_torque but found none"

        # Each gain write must be inside a ``with`` block within the method.
        with_ranges: list[tuple[int, int]] = []
        for node in ast.walk(method):
            if isinstance(node, ast.With):
                # Collect the line range of the with block's body.
                body_lines = [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")]
                if body_lines:
                    with_ranges.append((min(body_lines), max(body_lines)))

        for write in gain_writes:
            inside = any(lo <= write.lineno <= hi for lo, hi in with_ranges)
            assert inside, (
                f"gainprm/biasprm write at line {write.lineno} of set_torque is not inside a "
                f"'with' block (engine lock). Every model gain mutation must hold the engine's "
                f"lock to prevent a torn read from a concurrent mj_step."
            )

    def test_actuator_resolved_by_name_not_cached_index(self) -> None:
        """set_torque must resolve the actuator by name, not a connect-time index.

        A scene recompile (add_object, add_robot) reallocates the model and can
        shift actuator indices, so a cached ``actuator_id`` from connect time
        may address a different actuator after the recompile.
        """
        source = _source_path().read_text()
        method = _method_ast(source, "set_torque")
        assert method is not None

        source_text = ast.get_source_segment(
            _source_path().read_text(), method
        )
        assert source_text is not None

        # The old pattern used ``model.actuator(binding.actuator_id)`` which is index-based.
        # The fix should use name-based resolution (e.g. ``_named(model, "actuator", ...)``)
        # and NOT reference ``actuator_id`` for the gain write.
        assert "actuator_id" not in source_text, (
            "set_torque still references actuator_id for gain writes; it should resolve "
            "actuators by name via _named() to survive scene recompiles"
        )
