"""The mesh kill-switch predicate, as a leaf both ``core`` and ``session`` can import.

:func:`mesh_disabled_by_env` used to live in :mod:`strands_robots.mesh.core`.
:mod:`strands_robots.mesh.session` needed it too, at every door that reaches
``zenoh.open``, and reached it with an in-function import of ``core`` -- which
in turn imports ``session`` at module scope. That is an import cycle
(``core -> session -> core``, and ``core -> sensors -> session -> core``), and
CodeQL reports each edge of it. Moving the predicate one layer down removes the
upward edge instead of hiding it: ``session`` imports this module, ``core``
re-exports the name so every existing ``from strands_robots.mesh.core import
mesh_disabled_by_env`` keeps working, and this module imports only
:mod:`strands_robots._mesh_switch`, which imports nothing from the package.
"""

from __future__ import annotations

from strands_robots._mesh_switch import mesh_env_request

__all__ = ["mesh_disabled_by_env"]


def mesh_disabled_by_env() -> bool:
    """Report whether ``STRANDS_MESH`` forces the mesh off.

    ``STRANDS_MESH=false`` (or ``0`` / ``no``) is documented in README's
    Configuration table as "a hard kill switch that also overrides an explicit
    ``mesh=True``". An operator who sets it is asking for no Zenoh session and no
    presence on the fleet, so every path that can open one answers this -- not
    only :func:`strands_robots.mesh.core.init_mesh`.

    The switch is one-directional here: it only ever forces mesh OFF. Opting a
    bare ``Robot()`` *on* via ``STRANDS_MESH=true`` is resolved in the ``Robot``
    factory, which reads the affirmative spellings instead. A caller asking "may
    I start a mesh?" wants this predicate; a caller asking "was I asked to start
    one?" wants that one, and the two are not each other's negation -- an unset
    variable answers False to both.

    Resolved by :func:`strands_robots._mesh_switch.mesh_env_request`, which
    holds both halves of the vocabulary. That is what makes an unrecognized
    value reportable: this predicate alone cannot tell ``off`` (a typo) from
    ``true`` (the other reader's business), because both are equally "not a
    kill" to it.
    """
    return mesh_env_request() is False
