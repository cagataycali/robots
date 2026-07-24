"""Built-in ``Policy`` implementations must document every public method/property.

The :class:`~strands_robots.policies.base.Policy` ABC documents its runtime
contract (``get_actions`` / ``get_actions_sync`` / ``set_robot_state_keys`` /
``reset`` / ``set_control_frequency`` / ``set_rtc_observed_delay`` /
``requires_images`` / ``execution_horizon`` / ``is_chunk_emitting`` /
``provider_name``) with rich docstrings. The dependency-free built-in policies
that ship in the core package - :class:`~strands_robots.policies.mock.MockPolicy`
(the testing reference), :class:`~strands_robots.policies.composite.CompositePolicy`
(the lower/upper body split), and
:class:`~strands_robots.policies.persistent.PersistentPolicy` (the load-once
reusable handle) - override or delegate those members, and each override needs
its own docstring rather than silently leaning on the inherited one: a
``PersistentPolicy`` method that forwards to the wrapped policy still has
wrapper-specific behavior (locking, cache residency) worth stating.

This guard walks the core policy modules by AST (no import, so it never needs
any optional policy backend installed) and fails if any public method or
property of a policy class defines no docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.policies as policies_pkg

_PACKAGE_DIR = Path(policies_pkg.__file__).parent

# Core, dependency-free policy modules: the ABC plus the three built-in
# implementations that ship without any optional backend. Backend-specific
# providers (groot, cosmos3, lerobot_local, ...) live in subpackages guarded by
# optional deps and are out of scope for this import-free guard.
_CORE_MODULES = ("base.py", "mock.py", "composite.py", "persistent.py")

_EXPECTED_CLASSES = {
    "base.py::Policy",
    "base.py::ChunkedPolicy",
    "mock.py::MockPolicy",
    "composite.py::CompositePolicy",
    "persistent.py::PersistentPolicy",
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


def _policy_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every public class in the core modules."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _CORE_MODULES:
        source_file = _PACKAGE_DIR / module
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def test_core_modules_define_expected_policy_classes() -> None:
    """Guard: the scan actually found the built-in policy classes it protects."""
    assert set(_policy_classes()) == _EXPECTED_CLASSES, set(_policy_classes())


def test_builtin_policy_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _policy_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a built-in Policy class must have a "
        "docstring describing its behavior (the base ABC already does; a "
        "delegating wrapper still states its wrapper-specific behavior). "
        "Undocumented members: " + repr(offenders)
    )
