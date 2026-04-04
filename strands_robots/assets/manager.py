"""Robot Asset Manager for Strands Robots Simulation.

Resolves robot model files (MJCF XML) from:
    1. Bundled assets (``strands_robots/assets/`` — from MuJoCo Menagerie)
    2. Custom path (``STRANDS_ASSETS_DIR`` env var)
    3. User home (``~/.strands_robots/assets/``)
"""

import logging
import os
from pathlib import Path

from strands_robots.registry import (
    get_robot,
    list_robots,
)
from strands_robots.registry import (
    resolve_name as resolve_robot_name,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Asset directory resolution
# ─────────────────────────────────────────────────────────────────────

_BUNDLED_DIR = Path(__file__).parent
_USER_CACHE_DIR = Path.home() / ".strands_robots" / "assets"


def get_assets_dir() -> Path:
    """Get the primary assets directory (user cache).

    Returns ``~/.strands_robots/assets/`` by default (writable, not in pip package).
    Override with ``STRANDS_ASSETS_DIR`` env var.
    """
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom)
    else:
        d = _USER_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_search_paths() -> list[Path]:
    """Get ordered list of asset search paths.

    Order:
        1. User cache (``~/.strands_robots/assets/``)
        2. Custom path from ``STRANDS_ASSETS_DIR`` env var
        3. Bundled package dir (``strands_robots/assets/`` — XML only)
        4. CWD/assets
    """
    paths = []

    # User cache first (where downloads go)
    paths.append(get_assets_dir())

    # Custom path from STRANDS_ASSETS_DIR
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        for p in custom.split(":"):
            cp = Path(p)
            if cp not in paths:
                paths.append(cp)

    # Bundled directory (XML files only, no meshes in pip package)
    if _BUNDLED_DIR not in paths:
        paths.append(_BUNDLED_DIR)

    # CWD
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets not in paths:
        paths.append(cwd_assets)

    return paths


# ─────────────────────────────────────────────────────────────────────
# Model path resolution (delegates to registry)
# ─────────────────────────────────────────────────────────────────────


def _auto_download_robot(name: str, info: dict) -> bool:
    """Auto-download a single robot's assets via robot_descriptions.

    Called lazily when resolve_model_path finds XML but no meshes.
    Returns True if download succeeded.
    """
    try:
        # Lazy: download_assets depends on optional robot_descriptions package
        from strands_robots.tools.download_assets import (
            _download_from_github,
            _download_via_robot_descriptions,
            _robot_descriptions_available,
            get_user_assets_dir,
        )
    except ImportError:
        logger.warning("Auto-download unavailable: install robot_descriptions for automatic asset downloads")
        return False

    dest_dir = get_user_assets_dir()
    canonical = resolve_robot_name(name)

    # Try robot_descriptions first (covers most robots)
    if _robot_descriptions_available():
        results = _download_via_robot_descriptions({canonical: info}, dest_dir)
        if results.get(canonical, "").startswith("downloaded"):
            logger.info("Auto-downloaded %s via robot_descriptions", canonical)
            return True

    # Try custom GitHub source
    source = info.get("asset", {}).get("source", {})
    if source.get("type") == "github":
        result = _download_from_github(canonical, info, dest_dir)
        if result.startswith("downloaded"):
            logger.info("Auto-downloaded %s from GitHub", canonical)
            return True

    return False


def _has_meshes(directory: Path) -> bool:
    """Check if a directory tree contains mesh files."""
    _MESH_EXTS = {".stl", ".obj", ".msh", ".ply"}
    return any(f.suffix.lower() in _MESH_EXTS for f in directory.rglob("*") if f.is_file())


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
    for search_dir in get_search_paths():
        model_path = search_dir / asset_dir_name / xml_file
        if model_path.exists():
            candidates.append(model_path)

    if not candidates:
        # No XML found at all — try auto-download, then re-search
        logger.info("No XML found for %s, attempting auto-download...", name)
        if _auto_download_robot(name, info):
            for search_dir in get_search_paths():
                model_path = search_dir / asset_dir_name / xml_file
                if model_path.exists():
                    candidates.append(model_path)

    if not candidates:
        logger.warning("Robot model not found: %s → %s/%s", name, asset_dir_name, xml_file)
        return None

    # Prefer the candidate whose directory contains mesh files,
    # because an XML without meshes will fail to load in MuJoCo.
    for path in candidates:
        if _has_meshes(path.parent):
            logger.debug("Resolved %s → %s (has meshes)", name, path)
            return path

    # XML found but no meshes — auto-download and re-check
    logger.info("XML found for %s but no meshes, attempting auto-download...", name)
    if _auto_download_robot(name, info):
        # Re-scan after download (new symlinks may have appeared)
        for search_dir in get_search_paths():
            model_path = search_dir / asset_dir_name / xml_file
            if model_path.exists() and _has_meshes(model_path.parent):
                logger.debug("Resolved %s → %s (auto-downloaded)", name, model_path)
                return model_path

    # Final fallback: return first candidate (some robots have no meshes)
    logger.debug("Resolved %s → %s (no meshes available)", name, candidates[0])
    return candidates[0]


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
        dir_path = search_dir / asset_dir
        if dir_path.exists():
            return dir_path
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
