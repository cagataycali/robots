"""Sole owner of the ``STRANDS_MESH_BACKEND`` vocabulary.

Two modules resolve that variable and both must agree.
:func:`strands_robots.mesh.session._backend_choice` runs first, on every
session and publish path, and its verdict is what
``session._is_transport_backend()`` gates on;
:func:`strands_robots.mesh.transport.factory.get_transport` runs only once that
verdict is ``iot`` or ``bridge``. Each used to read the variable itself, and
they disagreed about the one case that matters: the factory reported an unknown
value and the gate in front of it did not, so a typo resolved to ``zenoh``, the
factory was never consulted, and the report could not fire for the input class
it named. ``STRANDS_MESH_BACKEND=iott`` produced a plain Zenoh session
indistinguishable from an explicit ``zenoh``.

This module holds the accepted values, the fallback and the report, so the two
readers cannot drift again. It imports nothing from the mesh package: the
factory reaches ``session`` through ``transport.zenoh_transport``, so a resolver
either reader had to import from the other would close that cycle, and a
call-time import inside ``session`` would raise in the documented "no zenoh
installed" case, where ``get_session`` must return ``None`` rather than
propagate an ``ImportError``.
"""

from __future__ import annotations

import contextvars
import logging
import os

logger = logging.getLogger(__name__)

#: The env var this module owns.
BACKEND_ENV_VAR = "STRANDS_MESH_BACKEND"

#: Transports :func:`strands_robots.mesh.transport.factory.get_transport` can
#: construct. Anything else is a typo.
BACKENDS = ("zenoh", "iot", "bridge")

#: Where an unset or unrecognized value lands.
DEFAULT_BACKEND = "zenoh"

#: Unknown values already reported. Keyed by the offending value, so a second
#: distinct typo is still news. A set mutated via ``.add`` (never reassigned)
#: avoids a ``global`` rebind, matching
#: ``strands_robots.simulation.predicates._RESOLUTION_WARNED``.
_UNKNOWN_WARNED: set[str] = set()

#: Constructor-arg override, active only for the duration of an
#: ``init_mesh(..., mesh_backend=...)`` call. A ``ContextVar`` rather than a
#: plain module-level variable so concurrent ``Robot(..., mesh_backend=...)``
#: calls on different threads or asyncio tasks do not stamp over each other's
#: choice: each execution context reads its own binding. Consumed by
#: :func:`select_backend`; managed by
#: :func:`~strands_robots.mesh.core.init_mesh` via ``push_backend_override`` /
#: ``pop_backend_override`` so the override never outlives the call site.
#:
#: When set (not ``None``), takes precedence over :data:`BACKEND_ENV_VAR`.
#: When ``None`` (the default), the env var resolution is unchanged, keeping
#: 100% back-compat with the historical single-owner env-var contract.
_BACKEND_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "strands_mesh_backend_override", default=None
)


def push_backend_override(backend: str | None) -> contextvars.Token[str | None]:
    """Install a constructor-arg override for :func:`select_backend`.

    Returns a token that :func:`pop_backend_override` uses to restore the
    previous binding, so nested overrides (Robot inside a Robot; init_mesh
    called from tests that already set one) unwind correctly.

    A ``None`` value is a request to clear the override for the scope, not to
    change it: passing ``None`` from a caller who did not opt into an override
    is the documented no-op.
    """
    if backend is None:
        return _BACKEND_OVERRIDE.set(None)
    normalized = backend.strip().lower()
    if normalized not in BACKENDS:
        # An explicit constructor arg is a caller mistake we can name, not a
        # typo in the operator environment: raise instead of falling back the
        # way env-var resolution does. Same message shape as the env-var
        # warning so an operator seeing both learns them together.
        raise ValueError(
            f"Unknown mesh_backend={backend!r}. Valid values: {', '.join(BACKENDS)}."
        )
    return _BACKEND_OVERRIDE.set(normalized)


def pop_backend_override(token: contextvars.Token[str | None]) -> None:
    """Restore the binding captured by :func:`push_backend_override`."""
    _BACKEND_OVERRIDE.reset(token)


def current_backend_override() -> str | None:
    """Return the active override, or ``None`` if none is installed."""
    return _BACKEND_OVERRIDE.get()


def select_backend() -> str:
    """Return the configured mesh transport, one of :data:`BACKENDS`.

    Resolution order (first hit wins):

    1. Constructor-arg override installed by
       :func:`push_backend_override` -- i.e. ``Robot(mesh_backend=...)`` or
       ``init_mesh(mesh_backend=...)``. Already validated at push time, so no
       re-validation or reporting is needed here.
    2. :data:`BACKEND_ENV_VAR` environment variable. Case and surrounding
       whitespace are normalised, so ``IOT`` and ``" iot "`` both select
       ``iot``. An unrecognized value falls back to :data:`DEFAULT_BACKEND`
       - the policy is to keep the mesh running rather than crash the host on
       a typo - and is reported once per distinct offending value. Once per
       value rather than once per call because the gate that consults this
       runs per published message: reporting every call would put one line
       per telemetry sample in the operator's log.
    3. :data:`DEFAULT_BACKEND` when no override and no env var.

    Returns:
        The selected backend name.
    """
    override = _BACKEND_OVERRIDE.get()
    if override is not None:
        return override
    raw = os.getenv(BACKEND_ENV_VAR, DEFAULT_BACKEND).strip().lower()
    if raw in BACKENDS:
        return raw
    if raw not in _UNKNOWN_WARNED:
        _UNKNOWN_WARNED.add(raw)
        logger.warning(
            "Unknown %s=%r - falling back to %r. Valid values: %s.",
            BACKEND_ENV_VAR,
            raw,
            DEFAULT_BACKEND,
            ", ".join(BACKENDS),
        )
    return DEFAULT_BACKEND
