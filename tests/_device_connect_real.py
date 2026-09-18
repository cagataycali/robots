"""The real ``device_connect_edge``, and every integration module put back after.

Five test modules need the genuine ``device_connect_edge`` while three others
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
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

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


def restore(held: dict[str, ModuleType], prefix: str = INTEGRATION_PREFIX) -> None:
    """Put *held* back, entry and parent attribute, dropping anything newer.

    Args:
        held: What :func:`held_modules` returned before the swap.
        prefix: The package the snapshot covers.
    """
    purge(prefix)
    sys.modules.update(held)
    for name, module in held.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)


def use_the_real_edge() -> dict[str, ModuleType]:
    """Swap the mocked ``device_connect_edge`` for the real one on disk.

    Returns:
        The integration modules displaced by the swap, for :func:`restore`.
    """
    held = held_modules()
    for name in _EDGE_MODULES:
        module = sys.modules.get(name)
        # A real module carries ``__file__``; a MagicMock stand-in does not.
        if module is not None and not hasattr(module, "__file__"):
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
