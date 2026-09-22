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

import subprocess
import sys
from pathlib import Path

from strands_robots.simulation.predicates import PREDICATE_REGISTRY, register_predicate

NAME = "a_predicate_only_this_module_knows"


def test_a_test_may_extend_the_registry() -> None:
    """Registering is legitimate - it is the documented extension point."""
    register_predicate(NAME, lambda: lambda _sim: True)
    assert NAME in PREDICATE_REGISTRY


def test_the_next_test_finds_the_registry_as_shipped() -> None:
    """The restore happened between the two cells, with no fixture in this file."""
    assert NAME not in PREDICATE_REGISTRY


def test_a_session_that_first_imports_the_registry_inside_a_test_keeps_the_shipped_set() -> None:
    """The baseline is the shipped set even when nothing imported it at collection.

    ``tests/test_fleet_emergency_evacuation.py`` imports the predicates module
    for the first time *inside* a test - its example registers
    ``evacuation_abort_within`` lazily - so a teardown that reads
    ``sys.modules`` at setup finds no module, takes an empty baseline and wipes
    the 30 shipped predicates for the rest of the process: every later cell dies
    on ``ValueError: Unknown predicate 'inside_region'``. Run in a subprocess
    because any session that collects *this* file has already imported the
    module, which is exactly what masks the empty-baseline path.
    """
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fleet_emergency_evacuation.py",
            "--no-cov",
            "-p",
            "no:randomly",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
