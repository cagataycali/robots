"""Robot Asset Manager for Strands Robots Simulation.

Resolves robot model files (MJCF XML) from:
    1. Bundled assets (``strands_robots/assets/`` — from MuJoCo Menagerie)
    2. Custom paths (``STRANDS_URDF_DIR`` / ``STRANDS_ASSETS_DIR`` env vars)
    3. User home (``~/.strands_robots/assets/``)

All robot definitions now live in ``registry/robots.json``.
This module provides path resolution and backward-compatible helpers.

Source: https://github.com/google-deepmind/mujoco_menagerie (Apache-2.0)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from strands_robots.registry import (
    format_robot_table,
    get_robot,
    list_aliases,
    list_robots,
    list_robots_by_category,
    resolve_name as resolve_robot_name,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Asset directory resolution
# ─────────────────────────────────────────────────────────────────────

_ASSETS_DIR = Path(__file__).parent


def get_assets_dir() -> Path:
    """Get the path to the bundled assets directory."""
    return _ASSETS_DIR


def get_search_paths() -> List[Path]:
    """Get ordered list of asset search paths."""
    paths = [_ASSETS_DIR]

    # Custom paths from env
    custom = os.getenv("STRANDS_URDF_DIR") or os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        for p in custom.split(":"):
            paths.append(Path(p))

    # User home
    paths.append(Path.home() / ".strands_robots" / "assets")

    # CWD
    paths.append(Path.cwd() / "assets")

    return paths


# ─────────────────────────────────────────────────────────────────────
# Model path resolution (delegates to registry)
# ─────────────────────────────────────────────────────────────────────


def resolve_model_path(
    name: str,
    prefer_scene: bool = False,
) -> Optional[Path]:
    """Resolve a robot name to its MJCF model XML path.

    Looks up the robot in ``registry/robots.json``, then searches
    the asset directories for the actual file.

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
    xml_file = asset["scene_xml"] if prefer_scene else asset["model_xml"]

    for search_dir in get_search_paths():
        model_path = search_dir / asset["dir"] / xml_file
        if model_path.exists():
            logger.debug("Resolved %s → %s", name, model_path)
            return model_path

    logger.warning("Robot model not found: %s → %s/%s", name, asset["dir"], xml_file)
    return None


def resolve_model_dir(name: str) -> Optional[Path]:
    """Resolve a robot name to its asset directory (containing XML + meshes).

    Args:
        name: Robot name (canonical or alias).

    Returns:
        Path to the robot's asset directory, or None if not found.
    """
    info = get_robot(name)
    if not info or "asset" not in info:
        return None

    asset_dir = info["asset"]["dir"]
    for search_dir in get_search_paths():
        dir_path = search_dir / asset_dir
        if dir_path.exists():
            return dir_path
    return None


def get_robot_info(name: str) -> Optional[Dict]:
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


def list_available_robots() -> List[Dict]:
    """List all available robot models with their info.

    Returns:
        List of dicts with name, description, joints, category, available, path.
    """
    robots = []
    for r in list_robots(mode="sim"):
        path = resolve_model_path(r["name"])
        info = get_robot(r["name"]) or {}
        robots.append({
            "name": r["name"],
            "description": r.get("description", ""),
            "joints": r.get("joints"),
            "category": r.get("category", ""),
            "dir": info.get("asset", {}).get("dir", ""),
            "available": path is not None,
            "path": str(path) if path else None,
        })
    return robots


__all__ = [
    "resolve_model_path",
    "resolve_model_dir",
    "resolve_robot_name",
    "get_robot_info",
    "list_available_robots",
    "list_robots_by_category",
    "list_aliases",
    "format_robot_table",
    "get_assets_dir",
    "get_search_paths",
]
