"""Simulation factory — create_simulation() and runtime backend registration.

Mirrors the policy factory pattern: JSON-driven defaults with runtime
override capability. Backends are lazy-loaded on first use.

Usage::

    from strands_robots.simulation import create_simulation

    # Default backend (MuJoCo)
    sim = create_simulation()

    # Explicit backend
    sim = create_simulation("mujoco", timestep=0.001)

    # Future backends
    sim = create_simulation("isaac", gpu_id=0)
    sim = create_simulation("newton")

    # Custom backend (runtime-registered)
    from strands_robots.simulation.factory import register_backend
    register_backend("my_sim", lambda: MySimBackend, aliases=["custom"])
    sim = create_simulation("custom")
"""

import logging
from collections.abc import Callable
from typing import Any

from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Built-in backend registry (lazy loaders — no imports at module load)
# ─────────────────────────────────────────────────────────────────────

_BUILTIN_BACKENDS: dict[str, tuple[str, str]] = {
    "mujoco": (
        "strands_robots.simulation.mujoco.simulation",
        "Simulation",
    ),
    # Future:
    # "isaac": ("strands_robots.simulation.isaac.simulation", "IsaacSimulation"),
    # "newton": ("strands_robots.simulation.newton.simulation", "NewtonSimulation"),
}

_BUILTIN_ALIASES: dict[str, str] = {
    "mj": "mujoco",
    "mjc": "mujoco",
    "mjx": "mujoco",
    # "isaac_sim": "isaac",
    # "isaacsim": "isaac",
    # "nvidia": "isaac",
}

DEFAULT_BACKEND = "mujoco"

# ─────────────────────────────────────────────────────────────────────
# Runtime registration (for user-defined backends not in built-ins)
# ─────────────────────────────────────────────────────────────────────

_runtime_registry: dict[str, Callable[[], type[SimEngine]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_backend(
    name: str,
    loader: Callable[[], type[SimEngine]],
    aliases: list[str] | None = None,
    force: bool = False,
) -> None:
    """Register a custom simulation backend at runtime.

    Use this to add backends without editing source code.

    Args:
        name: Backend identifier (e.g., ``"my_physics"``).
        loader: Zero-arg callable that returns the backend **class**
            (not instance). Called lazily on first ``create_simulation()``.
        aliases: Optional short names that resolve to ``name``.
        force: If False (default), raises ValueError when ``name`` or
            an alias is already registered. Set True to overwrite.

    Raises:
        ValueError: If ``name`` or an alias conflicts with an existing
            registration and ``force`` is False.

    Example::

        from strands_robots.simulation.factory import register_backend

        register_backend(
            "bullet",
            lambda: BulletSimulation,
            aliases=["pybullet", "pb"],
        )
        sim = create_simulation("bullet")
    """
    if not force:
        if name in _runtime_registry or name in _BUILTIN_BACKENDS:
            raise ValueError(
                f"Backend {name!r} already registered. Use force=True to overwrite."
            )
        if aliases:
            for alias in aliases:
                if alias in _BUILTIN_ALIASES:
                    raise ValueError(
                        f"Alias {alias!r} conflicts with built-in alias. "
                        f"Use force=True to overwrite."
                    )
                if alias in _runtime_aliases:
                    raise ValueError(
                        f"Alias {alias!r} already registered. "
                        f"Use force=True to overwrite."
                    )

    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name
    logger.debug("Registered simulation backend: %s (aliases=%s)", name, aliases)


def list_backends() -> list[str]:
    """List all available backend names (built-in + runtime-registered).

    Returns:
        Sorted list of unique backend identifiers and aliases.

    Example::

        >>> list_backends()
        ['mj', 'mjc', 'mjx', 'mujoco']
    """
    names: set[str] = set()
    names.update(_BUILTIN_BACKENDS.keys())
    names.update(_BUILTIN_ALIASES.keys())
    names.update(_runtime_registry.keys())
    names.update(_runtime_aliases.keys())
    return sorted(names)


def _resolve_name(backend: str) -> str:
    """Resolve aliases to canonical backend name."""
    # Runtime aliases first (user overrides win)
    if backend in _runtime_aliases:
        return _runtime_aliases[backend]
    # Built-in aliases
    if backend in _BUILTIN_ALIASES:
        return _BUILTIN_ALIASES[backend]
    return backend


def _import_backend_class(name: str) -> type[SimEngine]:
    """Import and return a backend class by canonical name."""
    # 1. Runtime registry (user-registered)
    if name in _runtime_registry:
        cls = _runtime_registry[name]()
        logger.debug("Loaded runtime backend: %s → %s", name, cls.__name__)
        return cls

    # 2. Built-in registry
    if name in _BUILTIN_BACKENDS:
        module_path, class_name = _BUILTIN_BACKENDS[name]
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        logger.debug("Loaded built-in backend: %s → %s.%s", name, module_path, class_name)
        return cls

    raise ValueError(f"Unknown simulation backend: {name!r}. Available: {', '.join(list_backends())}")


def create_simulation(
    backend: str = DEFAULT_BACKEND,
    **kwargs: Any,
) -> SimEngine:
    """Create a simulation backend instance.

    This is the primary entry point for creating simulations.
    Backend classes are lazy-loaded on first call.

    Args:
        backend: Backend name or alias. Defaults to ``"mujoco"``.
            Built-in: ``"mujoco"`` (aliases: ``"mj"``, ``"mjc"``, ``"mjx"``).
        **kwargs: Backend-specific keyword arguments passed to the
            constructor (e.g., ``tool_name``, ``timestep``).

    Returns:
        A ``SimEngine`` instance ready for ``create_world()``.

    Raises:
        ValueError: If the backend name is not recognized.
        ImportError: If the backend's dependencies are missing
            (e.g., ``pip install mujoco``).

    Examples::

        # Default (MuJoCo)
        sim = create_simulation()
        sim.create_world()
        sim.add_robot("so100")

        # With alias
        sim = create_simulation("mj")

        # Pass kwargs to backend constructor
        sim = create_simulation("mujoco", tool_name="my_sim")
    """
    canonical = _resolve_name(backend)
    logger.info("Creating simulation: %s (resolved from %r)", canonical, backend)

    BackendClass = _import_backend_class(canonical)
    return BackendClass(**kwargs)
