"""A fresh import that puts back every binding the import rebound.

``importlib.import_module`` binds a submodule in **two** places: the
``sys.modules`` entry, and an attribute of the same name on its parent package.
A cell that wants a module's top level to run again therefore has two things to
undo, and ``monkeypatch`` restores whichever ones it was told about. Told about
only the entry, the fresh module outlives the cell as ``parent.leaf`` while
``sys.modules`` holds the original, and the two paths stop naming the same
object for the rest of the session::

    monkeypatch.delitem(sys.modules, "a.b.c", raising=False)
    importlib.import_module("a.b.c")        # rebinds a.b.c AND a.b's attribute
    ...                                    # teardown restores only the entry

    import a.b.c as m                      # -> the fresh module (attribute)
    from a.b.c import f                    # -> the original's f (entry)
    monkeypatch.setattr(m, "f", double)    # patches the fresh module; the
                                           # production `from a.b.c import f`
                                           # still reaches the original

Which side a spelling lands on is not obvious, and both spellings are common in
this tree, so the split is silent: a stub goes on one module and the code under
test reads the other, and the cell passes while grading nothing.

Measured, with ``strands_robots.simulation.policy_runner`` split this way and
both ``except Exception`` guards deleted from
:func:`strands_robots.tools.run_policy._finalize_episode` - the very tolerance
those cells exist to pin:

===================================================  ======================
selection                                            outcome
===================================================  ======================
the two ``test_finalize_swallows_*`` cells alone      2 failed
the same two after the split is left behind          2 passed
===================================================  ======================

Reaching for this function instead of the two statements keeps the pair
together. It is the only place in the test trees that removes a ``sys.modules``
entry in order to import it again;
:mod:`tests.test_sys_modules_removal_leaves_no_orphan` grades that.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def reimport(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    """Run *name*'s top level again, and put both of its bindings back after.

    Args:
        monkeypatch: The cell's fixture, which performs the restoration at
            teardown. Both bindings are recorded on it before the import, so
            an assertion failure between here and teardown still restores.
        name: Dotted module path, e.g. ``"strands_robots.simulation.policy_runner"``.
            A top-level name has no parent package to rebind and only needs the
            entry.

    Returns:
        The freshly imported module - the object ``sys.modules`` holds until
        teardown, so a cell that wants to read the new copy has it without
        going back through an import spelling.
    """
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = importlib.import_module(parent_name)
        # ``raising=False`` is what covers a parent that does not carry the
        # attribute yet: monkeypatch records "absent" and deletes at teardown,
        # so a module nothing had imported is not left bound either.
        monkeypatch.setattr(parent, leaf, getattr(parent, leaf, None), raising=False)
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)
