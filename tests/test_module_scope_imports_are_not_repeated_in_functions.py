"""A name bound by a module-scope import is not imported again inside a function.

``import asyncio`` at the top of ``strands_robots/hardware_robot.py`` and
``import asyncio`` inside ``_run_control_loop`` bind the same object to the same
name; the second is a lookup in ``sys.modules`` that shadows a global with an
identical local and buys nothing - not laziness, since the module is already
loaded by the time the function is defined, and not isolation, since it is the
same module object. What it does cost is a reader: a function-scope import here
is the shape this package uses for a *deliberately* deferred dependency (a heavy
optional import with a documented reason, per AGENTS.md Key Conventions), so a
redundant one reads as a deferral that has a reason the reader then goes looking
for. The one above carried the comment ``# Import here to avoid conflicts``, and
there was no conflict to avoid.

CodeQL reports the shape as ``py/repeated-import``. Alert 67 sat open on
``refs/heads/main`` at that site from 2026-05-21, at note severity, gating
nothing - until three pull requests added lines above it on the same day and
CodeQL's positional baseline attributed the finding to each branch, opening a
``github-advanced-security`` review thread on every one. Under
``required_review_thread_resolution`` a thread gates the merge whatever the
severity, so a four-month-old finding on ``main`` held three approved branches.
That is the case PR Workflow step 8 names: a pre-existing alert fixed as its own
change on ``main`` rather than as a round on whichever branch shifted its line.

Measured on ``main`` at ``583dec56``: 13 such sites across 6 modules of the
package, each a bare repeat of a binding already made at module scope (``os``,
``numpy as np`` five times in one file, ``torch``, ``dataclasses``,
``importlib.util``, two ``from`` imports). ``tests/`` carries 151 on the same
predicate and is deliberately out of scope: there the shape is an idiom for
keeping a probe's imports beside the probe, it is graded by nothing this rule
protects, and a sweep that fires on a tenth of the test tree is boilerplate.

What is *not* a repeat, and what the controls below pin:

- A function-scope import of a name bound only under ``if TYPE_CHECKING:`` -
  that is the runtime half of a deferred import, and the module-scope half is
  not a binding at runtime.
- A function-scope import of a name a module-level ``try`` / ``except
  ImportError`` binds - the optional-dependency pattern, where the function
  re-import is what raises for a caller when the module-level one was refused.
- The same module bound to a *different* name (``import numpy`` at scope,
  ``import numpy as np`` in a function), or a different name from the same
  module (``from x import a`` at scope, ``from x import b`` in a function) -
  neither repeats a binding.

Only a statement that is a direct child of the module body is a module-scope
binding here; an import nested in any block is read as conditional and is not
one. That is what keeps the two exemptions above structural rather than listed.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap

import pytest

import strands_robots

_PACKAGE_ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent

_Binding = tuple[object, ...]


def _binding_keys(node: ast.Import | ast.ImportFrom) -> list[_Binding]:
    """One key per name an import statement binds, distinguishing the two forms.

    ``import a.b as c`` and ``from a import b as c`` bind different objects to
    ``c``, so the form is part of the key; so is the relative level, because
    ``from .x import y`` and ``from x import y`` name different modules.
    """
    if isinstance(node, ast.Import):
        return [("import", alias.name, alias.asname) for alias in node.names]
    return [("from", node.level, node.module, alias.name, alias.asname) for alias in node.names]


def _module_scope_bindings(tree: ast.Module) -> set[_Binding]:
    """Bindings made by import statements that are direct children of the module body."""
    keys: set[_Binding] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            keys.update(_binding_keys(stmt))
    return keys


def _function_scope_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every import statement whose nearest enclosing scope is a function."""
    found: list[ast.Import | ast.ImportFrom] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, True)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                if in_function:
                    found.append(child)
            else:
                visit(child, in_function)

    visit(tree, False)
    return found


def repeated_imports(source: str) -> list[tuple[int, str]]:
    """``(line, spelling)`` for each function-scope import repeating a module-scope binding."""
    tree = ast.parse(source)
    at_scope = _module_scope_bindings(tree)
    findings: list[tuple[int, str]] = []
    for node in _function_scope_imports(tree):
        repeated = [key for key in _binding_keys(node) if key in at_scope]
        if repeated:
            findings.append((node.lineno, ast.unparse(node)))
    return findings


def _package_modules() -> list[pathlib.Path]:
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def _module_id(path: pathlib.Path) -> str:
    return path.relative_to(_PACKAGE_ROOT).as_posix()


class TestNoFunctionRepeatsAModuleScopeImport:
    """The sweep over the installed package."""

    def test_the_package_is_walked(self) -> None:
        # A root that resolves to nothing would grade nothing and pass; pin the
        # walk against the one site the rule was written for.
        modules = _package_modules()
        assert _PACKAGE_ROOT / "hardware_robot.py" in modules
        assert len(modules) > 100

    @pytest.mark.parametrize("path", _package_modules(), ids=_module_id)
    def test_no_function_scope_import_repeats_a_module_scope_binding(self, path: pathlib.Path) -> None:
        findings = repeated_imports(path.read_text(encoding="utf-8"))
        assert not findings, (
            f"{_module_id(path)} imports inside a function a name it already binds at module scope; "
            "delete the function-scope import (the module is already loaded and bound), "
            "or if the module-scope binding is the one that should go, defer it with a documented reason:\n"
            + "\n".join(f"  line {line}: {spelling}" for line, spelling in findings)
        )


class TestThePredicateReadsBindingsNotNames:
    """Planted controls: what the sweep reports, and what it must leave alone."""

    def test_a_bare_repeat_of_an_import_is_reported(self) -> None:
        source = textwrap.dedent(
            """
            import asyncio

            def run() -> None:
                import asyncio
                asyncio.run(main())
            """
        )
        assert repeated_imports(source) == [(5, "import asyncio")]

    def test_a_repeat_of_an_aliased_import_is_reported_when_the_alias_matches(self) -> None:
        source = textwrap.dedent(
            """
            import numpy as np

            class Sim:
                def step(self) -> None:
                    import numpy as np
                    np.zeros(3)
            """
        )
        assert repeated_imports(source) == [(6, "import numpy as np")]

    def test_a_repeat_of_a_from_import_is_reported_even_beside_a_new_name(self) -> None:
        # ``predicate_reads_robot_base`` is new to the function; ``PREDICATE_REGISTRY``
        # is the repeat, and one repeated name is enough to report the statement.
        source = textwrap.dedent(
            """
            from pkg.predicates import PREDICATE_REGISTRY, make_predicate

            def collect() -> None:
                from pkg.predicates import PREDICATE_REGISTRY, predicate_reads_robot_base
            """
        )
        assert repeated_imports(source) == [
            (5, "from pkg.predicates import PREDICATE_REGISTRY, predicate_reads_robot_base")
        ]

    def test_a_repeat_inside_a_nested_function_or_async_def_is_reported(self) -> None:
        source = textwrap.dedent(
            """
            import os

            async def outer() -> None:
                def inner() -> None:
                    import os
                    os.getcwd()
            """
        )
        assert repeated_imports(source) == [(6, "import os")]

    def test_a_name_bound_only_under_type_checking_is_not_a_module_scope_binding(self) -> None:
        source = textwrap.dedent(
            """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from heavy import Thing

            def build() -> "Thing":
                from heavy import Thing
                return Thing()
            """
        )
        assert repeated_imports(source) == []

    def test_an_optional_dependency_guarded_at_module_scope_may_be_retried_in_a_function(self) -> None:
        source = textwrap.dedent(
            """
            try:
                import torch
            except ImportError:
                torch = None

            def to_device() -> None:
                import torch
                torch.zeros(1)
            """
        )
        assert repeated_imports(source) == []

    def test_the_same_module_under_a_different_alias_is_not_a_repeat(self) -> None:
        source = textwrap.dedent(
            """
            import numpy

            def f() -> None:
                import numpy as np
                np.zeros(1)
            """
        )
        assert repeated_imports(source) == []

    def test_a_different_name_from_the_same_module_is_not_a_repeat(self) -> None:
        source = textwrap.dedent(
            """
            from pathlib import PurePath

            def f() -> None:
                from pathlib import Path
                Path(".")
            """
        )
        assert repeated_imports(source) == []

    def test_a_relative_and_an_absolute_spelling_of_one_module_are_different_bindings(self) -> None:
        # The two resolve to one module inside a package, but the sweep reads
        # bindings and does not resolve modules, so it does not claim they repeat.
        source = textwrap.dedent(
            """
            from .processor import ProcessorBridge

            def f() -> None:
                from pkg.processor import ProcessorBridge
                ProcessorBridge()
            """
        )
        assert repeated_imports(source) == []

    def test_a_module_scope_import_repeated_at_module_scope_is_not_in_scope(self) -> None:
        # Two module-scope imports of one name is a different shape (and ruff's
        # domain); this sweep grades the function-scope repeat only.
        source = textwrap.dedent(
            """
            import os
            import os
            """
        )
        assert repeated_imports(source) == []
