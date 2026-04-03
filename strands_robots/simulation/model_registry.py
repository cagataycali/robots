"""Robot model resolution — URDF registry + Menagerie asset manager."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default URDF search paths (checked in order).
#
# Resolution order for legacy URDF lookups:
#   1. STRANDS_ASSETS_DIR (if set) — user override
#   2. ~/.strands_robots/assets/ — user cache
#   3. CWD/assets/ — project-local assets
#
# For new code, prefer resolve_model() which uses the Menagerie
# asset manager and falls back to these legacy paths.
_URDF_SEARCH_PATHS = [
    Path.home() / ".strands_robots" / "assets",
    Path.cwd() / "assets",
]

try:
    from strands_robots.assets import (  # noqa: I001
        format_robot_table as _format_robot_table,
        resolve_model_path as _resolve_menagerie_model,
    )

    _HAS_ASSET_MANAGER = True
except ImportError:
    _HAS_ASSET_MANAGER = False

try:
    from strands_robots.registry import get_robot as _get_robot
    from strands_robots.registry import resolve_name as _resolve_name

    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False

logger.debug("Asset manager available: %s", _HAS_ASSET_MANAGER)

# Legacy URDF registry — runtime cache for user-registered URDFs
_URDF_REGISTRY: dict[str, str] = {}

_ASSETS_DIR_OVERRIDE = os.getenv("STRANDS_ASSETS_DIR")
if _ASSETS_DIR_OVERRIDE:
    _URDF_SEARCH_PATHS.insert(0, Path(_ASSETS_DIR_OVERRIDE))

# Deprecated: STRANDS_URDF_DIR (use STRANDS_ASSETS_DIR)
_URDF_DIR_OVERRIDE = os.getenv("STRANDS_URDF_DIR")
if _URDF_DIR_OVERRIDE:
    _URDF_SEARCH_PATHS.insert(0, Path(_URDF_DIR_OVERRIDE))


def register_urdf(data_config: str, urdf_path: str):
    """Register a URDF/MJCF file for a data_config name."""
    _URDF_REGISTRY[data_config] = urdf_path
    logger.info("📋 Registered model for '%s': %s", data_config, urdf_path)


def resolve_model(name: str, prefer_scene: bool = True) -> str | None:
    """Resolve a robot name or data_config to an MJCF/URDF model path.

    Resolution order (local assets take priority):
    1. Legacy URDF registry (custom user registrations)
    2. URDF search paths (STRANDS_URDF_DIR, ./urdfs, CWD, etc.)
    3. Asset manager (Menagerie — fallback for standard robots)
    """
    # 1+2. Check local/custom paths first (user overrides win)
    local = resolve_urdf(name)
    if local:
        return local

    # 3. Fall back to asset manager (Menagerie)
    if _HAS_ASSET_MANAGER:
        path = _resolve_menagerie_model(name, prefer_scene=prefer_scene)
        if path and path.exists():
            return str(path)
        if prefer_scene:
            path = _resolve_menagerie_model(name, prefer_scene=False)
            if path and path.exists():
                return str(path)

    return None


def resolve_urdf(data_config: str) -> str | None:
    """Resolve a data_config name to a URDF file path (legacy)."""
    if data_config in _URDF_REGISTRY:
        urdf_rel = _URDF_REGISTRY[data_config]
        if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
            return urdf_rel
        for search_dir in _URDF_SEARCH_PATHS:
            candidate = search_dir / urdf_rel
            if candidate.exists():
                return str(candidate)

    if _HAS_REGISTRY:
        canonical = _resolve_name(data_config)
        info = _get_robot(canonical)
        if info and "legacy_urdf" in info:
            urdf_rel = info["legacy_urdf"]
            if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
                return urdf_rel
            for search_dir in _URDF_SEARCH_PATHS:
                candidate = search_dir / urdf_rel
                if candidate.exists():
                    return str(candidate)

    logger.debug("URDF not found for '%s' in search paths", data_config)
    return None


def list_registered_urdfs() -> dict[str, str | None]:
    """List all registered URDF mappings and their resolved paths."""
    return {config_name: resolve_urdf(config_name) for config_name in _URDF_REGISTRY}


def list_available_models() -> str:
    """List all available robot models (Menagerie + custom)."""
    if _HAS_ASSET_MANAGER:
        return _format_robot_table()

    lines = ["Registered URDFs:"]
    for name, path in _URDF_REGISTRY.items():
        resolved = resolve_urdf(name)
        status = "✅" if resolved else "❌"
        lines.append(f"  {status} {name}: {path}")
    return "\n".join(lines)
