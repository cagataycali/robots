"""The so101 reach task's command term implements every hook mjlab's CommandTerm requires.

``ReachCommand`` overrides ``_update_metrics``, ``_resample_command``,
``_update_command`` and ``_debug_vis_impl``: none is read inside this package
because mjlab's command manager calls them. This pins that the override set is
exactly the framework's abstract surface, so a hook renamed upstream (or a
typo here) surfaces as a red rather than as a task that instantiates and then
raises ``TypeError`` at the first reset.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import strands_robots

TASK = pathlib.Path(inspect.getfile(strands_robots)).parent / "training" / "mjlab_tasks" / "so101_reach.py"

#: The hooks mjlab's CommandTerm leaves abstract (mjlab 1.x), which is why the
#: task defines them without a reader of its own in this tree.
FRAMEWORK_HOOKS = frozenset({"_update_metrics", "_resample_command", "_update_command", "_debug_vis_impl"})


def _overrides(class_name: str) -> set[str]:
    tree = ast.parse(TASK.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_") and not n.name.startswith("__")
            }
    raise AssertionError(f"{class_name} not found in {TASK}")


def test_reach_command_defines_exactly_the_framework_hooks() -> None:
    private = _overrides("ReachCommand")
    assert private == FRAMEWORK_HOOKS, sorted(private ^ FRAMEWORK_HOOKS)


def test_the_hook_set_is_mjlab_s_abstract_surface() -> None:
    pytest.importorskip("mjlab", reason="mjlab (the [sim-mjlab] extra) is needed to read CommandTerm")
    from mjlab.managers.command_manager import CommandTerm

    abstract = set(getattr(CommandTerm, "__abstractmethods__", ()))
    assert abstract <= FRAMEWORK_HOOKS, sorted(abstract - FRAMEWORK_HOOKS)
    for hook in FRAMEWORK_HOOKS:
        assert hasattr(CommandTerm, hook), hook
