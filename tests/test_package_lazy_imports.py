"""Tests for the package-root lazy-import contract in ``strands_robots/__init__.py``.

``import strands_robots`` must stay cheap: heavy symbols (Robot, Simulation,
Gr00tPolicy, tools, ...) are resolved on first attribute access via PEP 562
``__getattr__``, while light symbols (Policy, MockPolicy, create_policy) import
eagerly. These tests pin the observable behavior of that loader:

- light symbols are importable with no extra dependencies,
- lazy symbols resolve, get cached, and return a stable identity,
- an unknown attribute raises ``AttributeError`` with the standard message,
- a lazy symbol whose backing module is missing warns and raises
  ``AttributeError`` chained from the original ``ImportError`` (so callers
  without an optional extra get a clear, recoverable failure).
"""

import ast
import warnings
from pathlib import Path

import pytest

import strands_robots


class TestEagerLightSymbols:
    """Light-weight policy symbols import without torch/lerobot/mujoco."""

    def test_policy_symbols_available_immediately(self):
        # These are real top-level imports, not lazy entries.
        assert "Policy" in strands_robots.__all__
        assert isinstance(strands_robots.MockPolicy, type)
        assert callable(strands_robots.create_policy)

    def test_light_symbols_are_not_lazy_entries(self):
        for name in ("Policy", "MockPolicy", "create_policy"):
            assert name not in strands_robots._LAZY_IMPORTS


class TestLazyResolution:
    """First attribute access resolves, caches, and returns stable identity."""

    def test_lazy_symbol_resolves_and_caches(self):
        # list_robots is registry-backed (no torch) so it always resolves.
        assert "list_robots" in strands_robots._LAZY_IMPORTS

        resolved = strands_robots.list_robots
        assert callable(resolved)

        # After first access the name is cached in the module dict so
        # __getattr__ is not invoked again.
        assert "list_robots" in vars(strands_robots)

        # Subsequent access returns the same object identity.
        assert strands_robots.list_robots is resolved

    def test_every_lazy_name_is_exported(self):
        # __all__ and _LAZY_IMPORTS must not drift apart: every lazy symbol
        # is part of the public surface.
        for name in strands_robots._LAZY_IMPORTS:
            assert name in strands_robots.__all__


class TestUnknownAttribute:
    """Unknown attributes raise AttributeError with the standard message."""

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            strands_robots.does_not_exist

    def test_dunder_attribute_raises_attributeerror(self):
        # Spurious dunder lookups (e.g. by copy/pickle) must not be swallowed.
        with pytest.raises(AttributeError):
            strands_robots.__wrapped__


class TestMissingDependencyContract:
    """A lazy symbol backed by an unimportable module warns then raises.

    Simulates the "optional extra not installed" path without uninstalling
    anything: register a temporary lazy entry pointing at a non-existent
    module, then assert the warn-and-raise contract.
    """

    def test_missing_module_warns_and_raises_chained_attributeerror(self):
        sentinel = "FakeMissingSymbolForTest"
        strands_robots._LAZY_IMPORTS[sentinel] = (
            "strands_robots._this_module_does_not_exist",
            "Thing",
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(AttributeError) as excinfo:
                    getattr(strands_robots, sentinel)

            # AttributeError carries the requested name and is chained from
            # the underlying ImportError so callers can introspect the cause.
            assert excinfo.value.args[0] == sentinel
            assert isinstance(excinfo.value.__cause__, ImportError)

            messages = [str(w.message) for w in caught]
            assert any(f"{sentinel} not available (missing dependencies)" in m for m in messages)
        finally:
            strands_robots._LAZY_IMPORTS.pop(sentinel, None)
            # The failed access must not leave a cached entry behind.
            assert sentinel not in vars(strands_robots)


class TestStaticExportContract:
    """Every ``__all__`` name must be statically resolvable.

    CodeQL's ``py/undefined-export`` (and most type-checkers) require that a
    name listed in ``__all__`` is defined in the module namespace by static
    analysis. The package resolves heavy symbols lazily via ``__getattr__``,
    which is invisible to a static analyzer, so each lazy name is also imported
    inside an ``if TYPE_CHECKING:`` block. This test pins that contract: a lazy
    symbol added to ``_LAZY_IMPORTS``/``__all__`` without a matching
    ``TYPE_CHECKING`` import would otherwise only be caught later by CodeQL.
    """

    @staticmethod
    def _type_checking_imported_names() -> set[str]:
        source = Path(strands_robots.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names: set[str] = set()
        for node in ast.walk(tree):
            # Match the top-level ``if TYPE_CHECKING:`` guard.
            if isinstance(node, ast.If):
                test = node.test
                is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                if not is_type_checking:
                    continue
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.ImportFrom):
                        for alias in stmt.names:
                            names.add(alias.asname or alias.name)
        return names

    def test_every_lazy_name_has_a_type_checking_import(self):
        type_checking_names = self._type_checking_imported_names()
        missing = sorted(name for name in strands_robots._LAZY_IMPORTS if name not in type_checking_names)
        assert not missing, (
            "Lazy symbols missing a TYPE_CHECKING import (CodeQL py/undefined-export "
            f"will flag these as exported-but-undefined): {missing}"
        )
