"""Robot model resolution - URDF registry + asset manager.

Bridges the robot registry with actual URDF/MJCF files on disk.

Resolution order for :func:`resolve_model`:
    1. User-registered URDFs (:func:`register_urdf`)
    2. URDF search paths (``STRANDS_ASSETS_DIR``, CWD, etc.)
    3. Asset manager (``robot_descriptions`` - fallback for standard robots)
"""

from __future__ import annotations

import logging
import os

from strands_robots.utils import get_search_paths

logger = logging.getLogger(__name__)

# URDF search paths are resolved lazily via :func:`strands_robots.utils.get_search_paths`
# at every lookup - this avoids snapshotting ``Path.cwd()`` and ``STRANDS_ASSETS_DIR``
# at import time, which caused silent wrong-path bugs when tests/notebooks chdir after
# import.

try:
    from strands_robots.assets import (
        format_robot_table,
        resolve_model_path,
    )

    _HAS_ASSET_MANAGER = True
except ImportError:
    _HAS_ASSET_MANAGER = False

try:
    from strands_robots.registry import get_robot, resolve_name

    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False

# Logged lazily on first resolution via _log_configuration_once() -
# avoids noisy INFO on every ``import strands_robots``.
_CONFIG_LOGGED = False


def _log_configuration_once() -> None:
    global _CONFIG_LOGGED
    if _CONFIG_LOGGED:
        return
    logger.debug("Asset manager available: %s", _HAS_ASSET_MANAGER)
    _CONFIG_LOGGED = True


# Runtime cache for user-registered URDFs
_URDF_REGISTRY: dict[str, str] = {}


def register_urdf(data_config: str, urdf_path: str) -> None:
    """Register a URDF/MJCF file for a data_config name."""
    _URDF_REGISTRY[data_config] = urdf_path
    logger.info("Registered model for '%s': %s", data_config, urdf_path)


def resolve_model(name: str, prefer_scene: bool = True) -> str | None:
    """Resolve a robot name or data_config to an MJCF/URDF model path.

    Resolution order (local assets take priority):
    1. User-registered URDFs (custom user registrations)
    2. URDF search paths (STRANDS_ASSETS_DIR, CWD, etc.)
    3. Asset manager (robot_descriptions - fallback for standard robots)
    """
    _log_configuration_once()

    def _try(candidate: str) -> str | None:
        # 1+2. Check local/custom paths first (user overrides win)
        local = resolve_urdf(candidate)
        if local:
            return local
        # 3. Fall back to asset manager
        if _HAS_ASSET_MANAGER:
            path = resolve_model_path(candidate, prefer_scene=prefer_scene)
            if path and path.exists():
                return str(path)
            if prefer_scene:
                path = resolve_model_path(candidate, prefer_scene=False)
                if path and path.exists():
                    return str(path)
        return None

    found = _try(name)
    if found:
        return found

    # Friction fix: callers (and LLMs) frequently guess a decorated variant
    # like "so101_default", "so101_sim", or "so101_robot" when the registry
    # key is just "so101". Strip a small set of common trailing qualifiers and
    # retry once before giving up, so the natural guess resolves instead of
    # forcing a list_urdfs round-trip.
    for suffix in ("_default", "_sim", "_robot", "_arm"):
        if name.endswith(suffix):
            stripped = name[: -len(suffix)]
            if stripped:
                found = _try(stripped)
                if found:
                    logger.info(
                        "resolve_model: '%s' not found; resolved via stripped name '%s'. Prefer the bare registry key.",
                        name,
                        stripped,
                    )
                    return found

    return None


def resolve_urdf(data_config: str) -> str | None:
    """Resolve a data_config name to a URDF file path.

    Also checks the registry's ``legacy_urdf`` field - a backward-compatible
    path for robots that were registered before the MJCF asset system
    was introduced (e.g. robots originally configured with raw URDF paths).
    """
    if data_config in _URDF_REGISTRY:
        urdf_rel = _URDF_REGISTRY[data_config]
        if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
            return str(urdf_rel)
        for search_dir in get_search_paths():
            candidate = search_dir / urdf_rel
            if candidate.exists():
                return str(candidate)

    if _HAS_REGISTRY:
        canonical = resolve_name(data_config)
        info = get_robot(canonical)
        # ``legacy_urdf``: backward-compatible URDF path from before the
        # MJCF asset system was introduced.  Kept so that existing
        # user configs referencing raw URDF paths continue to work.
        if info and "legacy_urdf" in info:
            urdf_rel = info["legacy_urdf"]
            if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
                return str(urdf_rel)
            for search_dir in get_search_paths():
                candidate = search_dir / urdf_rel
                if candidate.exists():
                    return str(candidate)

    logger.debug("URDF not found for '%s' in search paths", data_config)
    return None


def list_registered_urdfs() -> dict[str, str | None]:
    """List all registered URDF mappings and their resolved paths."""
    return {config_name: resolve_urdf(config_name) for config_name in _URDF_REGISTRY}


def _registered_urdf_lines() -> list[str]:
    """One ``[OK]`` / ``[MISSING]`` line per user-registered URDF.

    Single-sources the section :func:`list_available_models` appends, so the
    built-in-table branch and the asset-manager-absent branch cannot drift into
    two vocabularies for the same rows.

    Returns:
        One line per entry in the runtime registry, in registration order.
        Empty when nothing has been registered.
    """
    lines: list[str] = []
    for name, path in _URDF_REGISTRY.items():
        status = "[OK]" if resolve_urdf(name) else "[MISSING]"
        lines.append(f"{status} {name}: {path}")
    return lines


def list_available_models() -> str:
    """List all available robot models (Menagerie + custom).

    Both halves are always reported. The asset-manager table alone used to be
    returned whenever the asset manager was importable - which is every normal
    install - so a caller who had just registered an asset with
    :func:`register_urdf` was told by the discovery surface that it did not
    exist, while :func:`resolve_urdf` resolved it and ``add_robot`` spawned it.
    The registered section is omitted entirely when nothing is registered, so a
    default install's listing is unchanged.

    Returns:
        The built-in robot table, followed by a ``Registered URDFs:`` section
        when :func:`register_urdf` has been called. Without the asset manager
        only the registered section is available, so that is returned alone.
    """
    registered = _registered_urdf_lines()
    if _HAS_ASSET_MANAGER:
        table = str(format_robot_table())
        if registered:
            table += "\n\nRegistered URDFs:\n" + "\n".join(registered)
        return table

    return "\n".join(["Registered URDFs:", *registered])


def count_sim_robots() -> int:
    """Count available robot models in simulation registry.

    Useful for displaying available model count in status messages.
    Raises ImportError if the registry module is not available.
    """
    from strands_robots.registry import list_robots as _registry_list_robots

    return len(_registry_list_robots(mode="sim"))
