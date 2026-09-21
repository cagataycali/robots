"""A predicate a test registers is gone before the next test runs.

``PREDICATE_REGISTRY`` is process-global and :func:`register_predicate` is the
documented way to extend it, so a name one test adds is still there for every
later test in the process - and a grader that reads the registry as the set of
shipped predicates fails on a name only a test knows. Seven call sites used to
undo their own registration in a ``try``/``finally``; the session owns it now
(``tests/conftest.py::_predicate_registry_is_left_as_found``), and these two
cells - in this order, which is the order pytest runs them - are what says so.
"""

from __future__ import annotations

from strands_robots.simulation.predicates import PREDICATE_REGISTRY, register_predicate

NAME = "a_predicate_only_this_module_knows"


def test_a_test_may_extend_the_registry() -> None:
    """Registering is legitimate - it is the documented extension point."""
    register_predicate(NAME, lambda: lambda _sim: True)
    assert NAME in PREDICATE_REGISTRY


def test_the_next_test_finds_the_registry_as_shipped() -> None:
    """The restore happened between the two cells, with no fixture in this file."""
    assert NAME not in PREDICATE_REGISTRY
