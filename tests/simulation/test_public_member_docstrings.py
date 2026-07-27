"""Every public class defined in the top-level ``strands_robots.simulation``
modules must document each of its public methods and properties.

The simulation package root holds the backend-agnostic facades and value
objects the whole library funnels through: the :class:`SimEngine` ABC and its
concrete convenience layer (:mod:`~strands_robots.simulation.base`), the
policy-rollout runner and its typed config
(:mod:`~strands_robots.simulation.policy_runner`), the reward/predicate DSL
(:mod:`~strands_robots.simulation.predicates`), the benchmark surfaces, and the
backend factory. A reader driving those objects blind should learn what an
accessor returns from its docstring, not by reading the body -- for example
that :attr:`VideoConfig.enabled` reports whether an output path was configured,
or that :meth:`StatefulRewardTerm.reset` clears per-episode state at an episode
boundary. A rich class docstring does not document its individual accessors.

This mirrors the MuJoCo-backend guard
(``tests/simulation/mujoco/test_public_member_docstrings.py``) but scopes to the
simulation package's own top-level modules; the ``mujoco/``, ``isaac/``, and
``newton/`` backend subpackages carry (or will carry) their own guards and are
intentionally not walked here.

The check walks every top-level module by AST (no import, so it needs none of
the optional simulation backends installed) and fails if any public method or
property of a public class defines no docstring. Property setters and deleters
are exempt: a ``@x.setter`` mirrors its documented getter and a docstring on it
is noise, matching the convention elsewhere in the tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.simulation as simulation_pkg

_PACKAGE_DIR = Path(simulation_pkg.__file__).parent


def _is_setter_or_deleter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function is a ``@<name>.setter`` or ``@<name>.deleter``."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
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
        if _is_setter_or_deleter(node):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for public classes in top-level modules."""
    classes: dict[str, ast.ClassDef] = {}
    for source_file in sorted(_PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{source_file.name}::{node.name}"] = node
    return classes


def test_simulation_toplevel_has_public_classes() -> None:
    """Guard: the scan actually walked the top-level modules and found the anchors."""
    found = set(_public_classes())
    assert any(q.endswith("::VideoConfig") for q in found), found
    assert any(q.endswith("::StatefulRewardTerm") for q in found), found


def test_simulation_toplevel_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _public_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a public class in the top-level "
        "strands_robots.simulation modules must have a docstring describing "
        "what it returns or does (a rich class docstring does not document its "
        "individual accessors). Undocumented members: " + repr(offenders)
    )
