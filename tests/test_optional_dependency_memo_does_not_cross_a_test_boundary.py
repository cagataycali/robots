"""A stand-in for an optional dependency does not outlive the test that made it.

``strands_robots.utils.require_optional`` memoises what it resolves in a
process-global dict, so a test that rebinds ``sys.modules`` to stand in for a
module nothing installs leaves that stand-in behind: ``monkeypatch`` restores
the binding it was given, not the copy the package took. The session owns the
memo now (``tests/conftest.py::_optional_module_memo_holds_no_stand_in``), and
these two cells - in this order, which is the order pytest runs them - are what
says so.
"""

from __future__ import annotations

import sys
import types

import pytest

from strands_robots.utils import require_optional

NAME = "strands_robots_absent_optional_dependency"


def test_a_test_may_stand_in_for_an_absent_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standing in through ``sys.modules`` is legitimate - many tests do it."""
    stand_in = types.ModuleType(NAME)
    monkeypatch.setitem(sys.modules, NAME, stand_in)
    assert require_optional(NAME) is stand_in


def test_the_next_test_finds_the_dependency_absent_again() -> None:
    """The memo was emptied between the two cells, with no fixture in this file."""
    with pytest.raises(ImportError, match=NAME):
        require_optional(NAME)
