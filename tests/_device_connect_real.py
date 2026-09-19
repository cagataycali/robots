"""The real ``device_connect_edge``, and every integration module put back after.

Most Device Connect test modules need the genuine ``device_connect_edge``; two
install ``MagicMock`` stand-ins for it at collection time. Swapping the fakes out
is not enough on its own: ``strands_robots.device_connect.*`` has already been
imported against whichever base class was in place, so those modules have to go
too before the integration re-binds to the real ``@rpc`` / ``DeviceDriver``.

Dropping them is where the copies of this swap went wrong. A purge is not an
undo - it *orphans* every reference already bound to the module, and the
next import returns a different object::

    from strands_robots.device_connect import reachy_transport   # collection time
    ...
    monkeypatch.setattr(reachy_transport, "api", fake)           # patches the orphan
    driver.connect_eagerly()                                     # imports the fresh
                                                                 # copy - the real api

Measured, with ``tests/test_device_connect_hardening.py`` collected into the same
worker ahead of the reachy driver files: four cells in
``tests/drivers/test_reachy_wireless_daemon_protocol.py`` reported ``daemon
unreachable (reachy-a.local:8000)`` - a unit test resolving a hostname - and the
same four pass in the opposite order. That is one ordering, not a bug in either
file, which is why it only showed up once the suite was distributed across
workers.

So the swap here is scoped: it hands back what it displaced, and it hands back
*both* bindings an import makes - the ``sys.modules`` entry and the attribute of
the same name on the parent package. Restoring one of the two is worse than
restoring neither, because the two spellings then name different objects for the
rest of the session; :mod:`tests._module_reimport` documents that split for a
single module, and this module is the same rule over a package prefix.

The swap in the other direction has the same owner, because a stand-in installed
at collection time outlives the file that installed it. Collection of every file
precedes the teardown of any, so the second file to install a stand-in takes its
snapshot while the first file's is resident: the "original" edge it records is a
``MagicMock``, and the integration it records is bound to a class defined in a
test file. Handing either back at teardown serves the stand-in on to every later
import. Measured, with the two installing files selected together: after both
tore down, ``sys.modules["device_connect_edge"]`` was the first file's mock and
``robot_driver.DeviceDriver.__module__`` was that file's name. And a snapshot
taken *before* the swap does not cover a real module a later-collected sibling
imports during it: dropping that one at teardown is the orphaning above by
another route - measured as the same four reachy cells, with
``tests/test_device_connect_all_robots.py`` selected ahead of them. So a snapshot
records only what is real, and a restore hands back only what is real, drops what
is bound to a fake, and leaves a newer real module where the sibling that imported
it can still reach it.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any
from unittest.mock import NonCallableMock

#: The integration package a swap re-imports, and therefore has to put back.
INTEGRATION_PREFIX = "strands_robots.device_connect"

#: The ``device_connect_edge`` modules a sibling replaces with ``MagicMock``.
_EDGE_MODULES = (
    "device_connect_edge.drivers",
    "device_connect_edge.types",
    "device_connect_edge.device",
    "device_connect_edge",
)

#: The ones a swap re-imports eagerly, so the integration binds real classes.
_EDGE_REIMPORTS = ("device_connect_edge", "device_connect_edge.drivers", "device_connect_edge.types")

#: Where the stand-ins are defined: a class or function from under here is a fake.
_TESTS_DIR = os.path.dirname(os.path.realpath(__file__))


def _is_real(module: object) -> bool:
    """A real module carries ``__file__``; a ``MagicMock`` stand-in does not."""
    return module is not None and hasattr(module, "__file__")


@functools.cache
def _is_a_test_file(path: str) -> bool:
    return os.path.realpath(path).startswith(_TESTS_DIR + os.sep)


def _defined_in_a_test(origin: object) -> bool:
    """Whether the module named *origin* is one of the test files."""
    module = sys.modules.get(origin) if isinstance(origin, str) else None
    path = getattr(module, "__file__", None)
    return isinstance(path, str) and _is_a_test_file(path)


def _under(name: object, prefix: str) -> bool:
    return isinstance(name, str) and (name == prefix or name.startswith(f"{prefix}."))


def _holds_a_fake(namespace: Mapping[str, Any], prefix: str, seen: set[int]) -> bool:
    if id(namespace) in seen:
        return False
    seen.add(id(namespace))
    for value in list(namespace.values()):
        if isinstance(value, NonCallableMock):
            return True
        cls = value if isinstance(value, type) else type(value)
        if any(_defined_in_a_test(base.__module__) for base in cls.__mro__):
            return True
        if isinstance(value, FunctionType) and _under(value.__module__, prefix):
            if _holds_a_fake(value.__globals__, prefix, seen):
                return True
    return False


def bound_to_a_fake(module: ModuleType, prefix: str = INTEGRATION_PREFIX) -> bool:
    """Whether *module* was imported against a stand-in ``device_connect_edge``.

    A stand-in reaches an integration module in three shapes, and a module
    holding any of them was bound while a sibling's mock was resident: a
    ``Mock`` (``DeviceRuntime`` read off a ``MagicMock`` package), a class
    defined in a test file (the ``DeviceDriver`` base a mock file supplies, and
    so every driver class subclassing it), or a function from a sibling module
    under *prefix* that is itself bound so (the package caches
    ``init_device_connect`` from ``_impl`` on first access). A module that
    imports nothing from the edge - ``reachy_transport`` - holds none of them
    whichever edge was resident, and is real by this reading.

    A function defined in a test file is deliberately not one of the shapes.
    Every module that imports the edge imports ``DeviceDriver`` or
    ``DeviceRuntime`` beside the decorators, so the class or the ``Mock``
    already answers - and a test-defined function on a real module is the
    ``named_rpc_caller`` fixture's ``get_rpc_source_device`` patch, which can
    still be in place when a teardown reads the module.

    Args:
        module: An entry under *prefix* in ``sys.modules``.
        prefix: The package whose modules a function is followed into.

    Returns:
        ``True`` when a global of *module*, or of a function it imported from
        a sibling under *prefix*, came from a stand-in.
    """
    return _holds_a_fake(vars(module), prefix, set())


def held_modules(prefix: str = INTEGRATION_PREFIX) -> dict[str, ModuleType]:
    """Every module registered under *prefix*, to hand back later.

    Args:
        prefix: Dotted package path, matched exactly or as a parent.

    Returns:
        The ``sys.modules`` entries under *prefix*, as a plain snapshot.
    """
    return {name: module for name, module in sys.modules.items() if name == prefix or name.startswith(f"{prefix}.")}


def purge(prefix: str = INTEGRATION_PREFIX) -> None:
    """Drop every ``sys.modules`` entry under *prefix* so the next import runs."""
    for name in held_modules(prefix):
        del sys.modules[name]


def _bind(name: str, module: ModuleType) -> None:
    """Register *module* as *name*: the entry, and the attribute on its parent."""
    sys.modules[name] = module
    parent_name, _, leaf = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, leaf, module)


def _unbind(name: str, module: ModuleType) -> None:
    """Drop *module* from *name*: the entry, and the attribute on its parent.

    ``from pkg import leaf`` serves ``pkg.leaf`` when the attribute exists and
    imports only when it does not, so an entry dropped with its parent
    attribute left in place is still handed out by the second spelling.
    """
    del sys.modules[name]
    parent_name, _, leaf = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, leaf, None) is module:
        delattr(parent, leaf)


def _agree(prefix: str) -> None:
    """Make every package attribute under *prefix* name the registered module, or nothing.

    A package handed back from a snapshot carries the submodule attributes it
    had then, and one of those may name a module this restore dropped or did
    not hand back. Left as it is, ``from pkg import leaf`` serves that attribute
    while ``import pkg.leaf`` serves the registry - the split the module
    docstring describes.
    """
    for name, module in held_modules(prefix).items():
        for attr, value in list(vars(module).items()):
            if not isinstance(value, ModuleType) or value.__name__ != f"{name}.{attr}":
                continue
            registered = sys.modules.get(value.__name__)
            if registered is None:
                delattr(module, attr)
            elif registered is not value:
                setattr(module, attr, registered)


def restore(held: dict[str, ModuleType], prefix: str = INTEGRATION_PREFIX) -> None:
    """Put back what is real in *held*, and drop what is bound to a fake.

    Each name under *prefix*, in the snapshot or registered now, gets one of
    three answers. A snapshot that is real goes back, entry and parent attribute
    both, over whatever is registered now: it is the object every module
    collected before the swap still holds. A snapshot bound to a fake - taken
    while a sibling's stand-in was resident - is not handed back, and a newer
    entry bound to one is dropped, so the next import binds against the edge
    resident then rather than being served the stand-in. A newer entry that is
    real is left alone: a sibling collected after the snapshot holds it, and
    dropping it hands that sibling an orphan.

    Args:
        held: What :func:`held_modules` returned before the swap.
        prefix: The package the snapshot covers.
    """
    for name in sorted(set(held) | set(held_modules(prefix))):
        kept = held.get(name)
        if kept is not None and not bound_to_a_fake(kept, prefix):
            _bind(name, kept)
            continue
        current = sys.modules.get(name)
        if current is not None and bound_to_a_fake(current, prefix):
            _unbind(name, current)
    _agree(prefix)


def use_the_real_edge() -> dict[str, ModuleType]:
    """Swap the mocked ``device_connect_edge`` for the real one on disk.

    Returns:
        The integration modules displaced by the swap, for :func:`restore`.
    """
    held = held_modules()
    for name in _EDGE_MODULES:
        module = sys.modules.get(name)
        if module is not None and not _is_real(module):
            del sys.modules[name]
    for name in _EDGE_REIMPORTS:
        importlib.import_module(name)
    purge()
    return held


@contextlib.contextmanager
def real_device_connect_edge() -> Iterator[None]:
    """Run the block against the real edge package, then undo both bindings."""
    held = use_the_real_edge()
    try:
        yield
    finally:
        restore(held)


@dataclass(frozen=True)
class MockedEdge:
    """What :func:`use_a_mock_edge` displaced, for :func:`restore_the_edge`."""

    #: The real ``device_connect_edge`` module under each name, or ``None`` when
    #: none was registered - a sibling's stand-in is recorded as none.
    edge: dict[str, ModuleType | None]
    #: The integration modules registered at the swap.
    integration: dict[str, ModuleType]


def use_a_mock_edge(stand_ins: Mapping[str, Any]) -> MockedEdge:
    """Install *stand_ins* under the ``device_connect_edge`` names.

    The snapshot taken first records only what is real. A stand-in already
    registered under one of the names belongs to a sibling collected earlier,
    and recording it as the original would hand it back at teardown, after
    that sibling has already taken its own away.

    Args:
        stand_ins: One replacement per name in ``_EDGE_MODULES``.

    Returns:
        The swap, for :func:`restore_the_edge` at ``teardown_module``.
    """
    edge: dict[str, ModuleType | None] = {}
    for name in _EDGE_MODULES:
        module = sys.modules.get(name)
        edge[name] = module if _is_real(module) else None
    integration = held_modules()
    for name in _EDGE_MODULES:
        sys.modules[name] = stand_ins[name]
    return MockedEdge(edge, integration)


def restore_the_edge(swap: MockedEdge) -> None:
    """Undo :func:`use_a_mock_edge`: the edge names first, then the integration.

    A real module registered under one of the names now is never displaced: a
    sibling may already be bound to it, and the snapshot's answer for that name
    is at best the same object. A stand-in is taken away whichever file
    installed it, so the real package is importable for whatever runs next -
    the two installing files already did this to each other, and a file whose
    tests reach the real ``device_connect_agent_tools`` mid-run leans on it.

    Args:
        swap: What :func:`use_a_mock_edge` returned.
    """
    for name, original in swap.edge.items():
        if _is_real(sys.modules.get(name)):
            continue
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    restore(swap.integration)
