"""Robot Asset Manager for Strands Robots Simulation.

Resolves robot model files (MJCF XML) from:
    1. ``STRANDS_ASSETS_DIR`` env var (user override)
    2. User cache (``~/.strands_robots/assets/``)
    3. ``robot_descriptions`` package (MuJoCo Menagerie)
    4. Project-local ``./assets/``
"""

import logging
from pathlib import Path

from strands_robots.registry import (
    get_robot,
    list_robots,
)
from strands_robots.registry import (
    resolve_name as resolve_robot_name,
)
from strands_robots.utils import get_search_paths, safe_join

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Model path resolution (delegates to registry)
# ─────────────────────────────────────────────────────────────────────


def _auto_download_robot(name: str, info: dict) -> bool:
    """Delegate to :func:`strands_robots.assets.download.auto_download_robot`.

    Lazy import at call time is retained only to keep ``manager.py`` importable
    in environments where the optional ``robot_descriptions`` package (and its
    transitive heavyweight deps) are not installed.
    """
    try:
        from .download import auto_download_robot
    except ImportError:
        logger.warning("Auto-download unavailable: install strands-robots[sim] for automatic asset downloads")
        return False
    return auto_download_robot(name, info)


def _has_meshes(directory: Path) -> bool:
    """Check if a directory tree contains mesh files."""
    _MESH_EXTS = {".stl", ".obj", ".msh", ".ply"}
    return any(f.suffix.lower() in _MESH_EXTS for f in directory.rglob("*") if f.is_file())


def _resolve_candidates(asset_dir_name: str, xml_file: str, name: str) -> list[Path]:
    """Resolve candidate paths for a robot XML, with path-traversal protection.

    Uses ``_safe_join`` to prevent ``../`` in registry-sourced ``asset_dir_name``
    or ``xml_file`` from escaping the search directories.
    """
    candidates: list[Path] = []
    for search_dir in get_search_paths():
        try:
            model_path = safe_join(search_dir, f"{asset_dir_name}/{xml_file}")
        except ValueError:
            logger.warning("Path traversal attempt blocked for robot: %s", name)
            return []
        if model_path.exists():
            candidates.append(model_path)
    return candidates


def resolve_model_path(
    name: str,
    prefer_scene: bool = False,
) -> Path | None:
    """Resolve a robot name to its MJCF model XML path.

    Looks up the robot in ``registry/robots.json``, then searches
    the asset directories for the actual file.  If XML is found but
    mesh files are missing, automatically downloads them via
    ``robot_descriptions`` before returning.

    Args:
        name: Robot name (canonical or alias).
        prefer_scene: If True, return scene XML (with ground/lights)
                      instead of bare model XML.

    Returns:
        Path to the MJCF XML file, or None if not found.

    Examples::

        resolve_model_path("so100")             # → .../trs_so_arm100/so_arm100.xml
        resolve_model_path("so100", prefer_scene=True)  # → .../trs_so_arm100/scene.xml
        resolve_model_path("franka")            # → .../franka_emika_panda/panda.xml
    """
    info = get_robot(name)
    if not info or "asset" not in info:
        logger.warning("Unknown robot or no asset: %s", name)
        return None

    asset = info["asset"]
    # Explicit str() casts: dict subscript returns Any, but Path / Any → Any
    xml_file: str = str(asset["scene_xml"] if prefer_scene else asset["model_xml"])
    asset_dir_name: str = str(asset["dir"])

    candidates: list[Path] = []

    # Check user-registered asset path first (highest priority)
    user_path = info.get("_user_asset_path")
    if user_path:
        user_model = Path(user_path) / xml_file
        if user_model.exists():
            candidates.append(user_model)

    # Search standard paths with traversal protection
    candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates:
        # No XML found at all — try auto-download, then re-search
        logger.info("No XML found for %s, attempting auto-download...", name)
        if _auto_download_robot(name, info):
            candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates:
        logger.warning("Robot model not found: %s → %s/%s", name, asset_dir_name, xml_file)
        return None

    # Prefer the candidate whose directory contains mesh files,
    # because an XML without meshes will fail to load in MuJoCo.
    for path in candidates:
        if _has_meshes(path.parent):
            logger.debug("Resolved %s → %s (has meshes)", name, path)
            return Path(path)

    # XML found but no meshes — auto-download and re-check
    logger.info("XML found for %s but no meshes, attempting auto-download...", name)
    if _auto_download_robot(name, info):
        # Re-scan after download (new symlinks may have appeared)
        refreshed = _resolve_candidates(asset_dir_name, xml_file, name)
        for path in refreshed:
            if _has_meshes(path.parent):
                logger.debug("Resolved %s → %s (auto-downloaded)", name, path)
                return Path(path)

    # Final fallback: return first candidate (some robots have no meshes)
    logger.debug("Resolved %s → %s (no meshes available)", name, candidates[0])
    return Path(candidates[0])


def resolve_model_dir(name: str) -> Path | None:
    """Resolve a robot name to its asset directory (containing XML + meshes).

    Args:
        name: Robot name (canonical or alias).

    Returns:
        Path to the robot's asset directory, or None if not found.
    """
    info = get_robot(name)
    if not info or "asset" not in info:
        return None

    asset_dir: str = str(info["asset"]["dir"])
    for search_dir in get_search_paths():
        try:
            dir_path = safe_join(search_dir, asset_dir)
        except ValueError:
            logger.warning("Path traversal attempt blocked in resolve_model_dir: %s", asset_dir)
            return None
        if dir_path.exists():
            return Path(dir_path)
    return None


def get_robot_info(name: str) -> dict | None:
    """Get information about a robot model.

    Args:
        name: Robot name (canonical or alias).

    Returns:
        Dict with description, category, joints, asset info, etc.
    """
    info = get_robot(name)
    if info is None:
        return None
    result = dict(info)
    result["canonical_name"] = resolve_robot_name(name)
    path = resolve_model_path(name)
    result["resolved_path"] = str(path) if path else None
    result["available"] = path is not None
    return result


def list_available_robots() -> list[dict]:
    """List all available robot models with their info.

    Returns:
        List of dicts with name, description, joints, category, available, path.
    """
    robots = []
    for r in list_robots(mode="sim"):
        path = resolve_model_path(r["name"])
        info = get_robot(r["name"]) or {}
        robots.append(
            {
                "name": r["name"],
                "description": r.get("description", ""),
                "joints": r.get("joints"),
                "category": r.get("category", ""),
                "dir": info.get("asset", {}).get("dir", ""),
                "available": path is not None,
                "path": str(path) if path else None,
            }
        )
    return robots
