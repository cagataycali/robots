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

# Module-level conditional import - keeps manager.py importable in
# environments where the optional ``robot_descriptions`` package (and its
# transitive heavyweight deps like ``GitPython``) are not installed.
# When ``download`` is not available, auto-download simply returns False.
try:
    from .download import auto_download_robot as _auto_download_robot_impl
except ImportError:
    _auto_download_robot_impl = None  # type: ignore[assignment]


def _lookup(name: str, *, allow_discovery: bool = True) -> dict | None:
    """Resolve a registry entry for *name*, with ``robot_descriptions`` fallback.

    A curated ``robots.json`` entry always wins. When a name is unknown to the
    curated registry, fall back to :func:`strands_robots.registry.discovery`,
    which synthesizes an entry for any MJCF-capable ``robot_descriptions`` robot
    (the long tail: ``gen3``, ``iiwa14``, ``viper``, ...). Discovery imports the
    description module, which can trigger an asset download on first use, so
    this helper is used only by download-capable resolvers - never by the
    side-effect-free presence check.

    Args:
        name: Robot name (canonical or alias).
        allow_discovery: When False, answer from the curated registry alone. A
                         resolver whose caller declined the download must pass
                         False: the discovery import *is* the fetch, because
                         ``robot_descriptions`` calls ``clone_to_cache`` at
                         module scope, so consulting it would perform the exact
                         download the caller declined - including on the miss
                         path, where the answer is ``None`` either way.

    Returns:
        A registry-style entry, or ``None`` when the name is unknown (and, with
        ``allow_discovery=False``, when only discovery could have synthesized one).
    """
    info = get_robot(name)
    if info is not None:
        return info
    if not allow_discovery:
        return None
    from strands_robots.registry.discovery import discover_robot

    return discover_robot(name)


#
# Model path resolution (delegates to registry)
#


def _auto_download_robot(name: str, info: dict) -> bool:
    """Delegate to :func:`strands_robots.assets.download.auto_download_robot`.

    Returns ``False`` immediately when the download module is unavailable
    (e.g. ``robot_descriptions`` not installed).
    """
    if _auto_download_robot_impl is None:
        logger.warning("Auto-download unavailable: install strands-robots[sim-mujoco] for automatic asset downloads")
        return False
    return _auto_download_robot_impl(name, info)


def _model_meshes_resolve(model_path: Path) -> bool:
    """Whether every mesh *model_path* declares is on disk where MuJoCo looks.

    Asked through the single owner of that question,
    :func:`strands_robots.assets.download._mjcf_missing_meshes`, rather than by
    walking the model directory for files with a mesh extension. The two are not
    the same reading and they disagree in both directions:

    * ``<compiler meshdir="../meshes"/>`` places a model's meshes OUTSIDE its own
      directory - a real shipped layout - so a downward walk reports a complete
      asset as mesh-less. The resolver then reaches for a download that cannot
      change the answer, on every call, for a model that already loads.
    * a model missing one of the meshes it declares still has the rest of them on
      disk, so a walk reports it as fine and the fetch that would repair it is
      never attempted. The caller gets a path MuJoCo refuses to load.

    One owner is what keeps this resolver, :func:`~strands_robots.assets.download._needs_download`
    and :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._ensure_meshes`
    from judging the same model present and absent.

    Args:
        model_path: Path to the model's MAIN file.

    Returns:
        True when the model declares no meshes or every reference resolves - the
        two cases a caller may treat alike, since neither has anything to fetch.
        False when a declared reference is absent, and when the model cannot be
        read: this resolver's own next step is a download, which is the reading
        :func:`~strands_robots.assets.download._needs_download` takes of the same
        failure, because a fetch is what replaces a file it cannot read.
    """
    from strands_robots.assets.download import _mjcf_missing_meshes

    try:
        return not _mjcf_missing_meshes(model_path)
    except (OSError, UnicodeDecodeError):
        return False


def _resolve_candidates(asset_dir_name: str, xml_file: str, name: str) -> list[Path]:
    """Resolve candidate paths for a robot XML, with path-traversal protection.

    Uses ``safe_join`` to prevent ``../`` in registry-sourced ``asset_dir_name``
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


def is_robot_asset_present(name: str) -> bool:
    """Check whether a robot's model XML exists on disk without triggering downloads.

    Pure filesystem check - no auto-download, no mesh walk, no network.
    Use this for status queries (e.g. ``download_assets(action="status")``)
    where you need to quickly check presence without side effects.

    Args:
        name: Robot name (canonical or alias).

    Returns:
        True if the model XML file exists on at least one search path.
    """
    info = get_robot(name)
    if not info or "asset" not in info:
        return False

    asset = info["asset"]
    xml_file: str = str(asset["model_xml"])
    asset_dir_name: str = str(asset["dir"])

    # Check user-registered path first
    user_path = info.get("_user_asset_path")
    if user_path:
        try:
            user_model = safe_join(Path(user_path), xml_file)
            if user_model.exists():
                return True
        except ValueError:
            pass

    # Check standard search paths
    for search_dir in get_search_paths():
        try:
            model_path = safe_join(search_dir, f"{asset_dir_name}/{xml_file}")
            if model_path.exists():
                return True
        except ValueError:
            continue

    return False


def resolve_model_path(
    name: str,
    prefer_scene: bool = False,
    *,
    allow_download: bool = True,
) -> Path | None:
    """Resolve a robot name to its MJCF model XML path.

    Looks up the robot in ``registry/robots.json``, then searches
    the asset directories for the actual file.  If no XML is found, or a mesh the
    model DECLARES is absent, downloads the asset via ``robot_descriptions``
    before returning - unless ``allow_download`` declines.

    Mesh presence is read from the model's declarations, resolved against its
    ``<compiler meshdir>`` the way MuJoCo resolves them, so a model whose meshes
    live outside its own directory is not mistaken for one that has none.

    ``allow_download=False`` makes this a pure filesystem lookup: no network, no
    ``robot_descriptions`` import. Within the curated registry it is exactly
    equivalent to a download that declines, so it cannot change the answer for an
    asset already on disk - an absent XML still returns ``None`` and an XML whose
    declared mesh is absent still returns its first candidate, the same two
    outcomes a failed download produces today.

    It declines the discovery fallback too, so a name only
    :mod:`strands_robots.registry.discovery` could synthesize an entry for reports
    a miss instead of importing a description module to find out. That import is
    itself the fetch - ``robot_descriptions`` calls ``clone_to_cache`` at module
    scope - so consulting discovery clones the upstream asset repository on a cold
    cache, including on the miss path where the answer is ``None`` either way.

    Prefer it in any caller that reports on assets rather than loading them.
    :func:`is_robot_asset_present` answers *whether* an asset is on disk without a
    download; this answers *where* on the same terms and over the same names -
    both read the curated registry alone.

    Args:
        name: Robot name (canonical or alias).
        prefer_scene: If True, return scene XML (with ground/lights)
                      instead of bare model XML.
        allow_download: When False, never attempt an asset download and never
                        consult discovery; resolve from what is already on disk
                        for a curated name, or report a miss.

    Returns:
        Path to the MJCF XML file, or None if not found.

    Examples::

        resolve_model_path("so100")             # → .../trs_so_arm100/so_arm100.xml
        resolve_model_path("so100", prefer_scene=True)  # → .../trs_so_arm100/scene.xml
        resolve_model_path("franka")            # → .../franka_emika_panda/panda.xml
    """
    info = _lookup(name, allow_discovery=allow_download)
    if not info or "asset" not in info:
        # DEBUG, not WARNING: resolve_model() probes several candidate names
        # (incl. suffix-stripped variants) and handles a None return cleanly,
        # so a miss here is normal control flow -- not something the user
        # needs to see on every add_robot.
        logger.debug("Unknown robot or no asset: %s", name)
        return None

    asset = info["asset"]
    # Explicit str() casts: dict subscript returns Any, but Path / Any → Any
    xml_file: str = str(asset["scene_xml"] if prefer_scene else asset["model_xml"])
    asset_dir_name: str = str(asset["dir"])

    candidates: list[Path] = []

    # Check user-registered asset path first (highest priority).
    # ``xml_file`` comes from user_robots.json, so we still gate it through
    # :func:`safe_join` to block path traversal even for user-authored entries
    # (defense in depth - protects against a compromised user_robots.json and
    # keeps the trust boundary identical to the built-in registry path).
    user_path = info.get("_user_asset_path")
    if user_path:
        try:
            user_model = safe_join(Path(user_path), xml_file)
        except ValueError:
            logger.warning(
                "Path traversal blocked in _user_asset_path for %s: %r",
                name,
                xml_file,
            )
            user_model = None
        if user_model is not None and user_model.exists():
            candidates.append(user_model)

    # Search standard paths with traversal protection
    candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates and allow_download:
        # No XML found at all - try auto-download, then re-search
        logger.info("No XML found for %s, attempting auto-download...", name)
        if _auto_download_robot(name, info):
            candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates:
        logger.warning("Robot model not found: %s -> %s/%s", name, asset_dir_name, xml_file)
        return None

    # Prefer the candidate whose declared meshes all resolve, because an XML with
    # a mesh reference MuJoCo cannot open will fail to load. Each candidate is
    # read on its own, so the download below cannot be answered from a
    # pre-download reading.
    for path in candidates:
        if _model_meshes_resolve(path):
            logger.debug("Resolved %s -> %s (declared meshes resolve)", name, path)
            return Path(path)

    # XML found but no meshes - auto-download and re-check. Declining the
    # download leaves the first candidate to the fallback below, which is what a
    # download that fails already does.
    if allow_download:
        logger.info("XML found for %s but a declared mesh is absent, attempting auto-download...", name)
        if _auto_download_robot(name, info):
            # Re-read after download: the fetch is exactly the change the pass
            # above could not have observed.
            refreshed = _resolve_candidates(asset_dir_name, xml_file, name)
            for path in refreshed:
                if _model_meshes_resolve(path):
                    logger.debug("Resolved %s -> %s (auto-downloaded)", name, path)
                    return Path(path)

    # Final fallback: return the first candidate, which is what a download that
    # cannot supply the absent reference leaves the caller with either way.
    logger.debug("Resolved %s -> %s (no meshes available)", name, candidates[0])
    return Path(candidates[0])


def resolve_model_dir(name: str, *, allow_download: bool = True) -> Path | None:
    """Resolve a robot name to its asset directory (containing XML + meshes).

    This resolver reads the filesystem. It returns a directory that already
    exists on a search path and never downloads the asset itself, so the only
    call here that can reach the network is the registry lookup: :func:`_lookup`
    falls back to ``robot_descriptions`` discovery for a name the curated
    registry does not know, and that import *is* the fetch.

    That left this resolver holding the side effect without the capability.
    ``discover_robot`` is documented "Call only from asset-resolution paths that
    are allowed to download", and this path downloads nothing, yet its caller had
    no way to decline the clone the way :func:`resolve_model_path`'s caller can.
    ``allow_download=False`` closes that gap and makes this a pure filesystem
    lookup - no network, no ``robot_descriptions`` import.

    The default stays open, so the long tail still resolves for a caller about to
    load a model. Declining costs the answer only for a robot that discovery
    alone can name; a curated robot never reaches the fallback either way.

    Args:
        name: Robot name (canonical or alias).
        allow_download: When False, answer from the curated registry alone
            rather than letting the discovery fallback clone the upstream asset
            repository.

    Returns:
        Path to the robot's asset directory, or None if not found.
    """
    info = _lookup(name, allow_discovery=allow_download)
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
    info = _lookup(name)
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

    Filesystem-only: :func:`is_robot_asset_present` for the availability flag
    and ``allow_download=False`` for the path, so listing what is installed never
    fetches what is not. The presence check alone was not enough - it stops the
    resolver reaching for an absent XML, but a cached XML whose meshes are missing
    passes it and was still fetched, once per such robot per listing.

    Returns:
        List of dicts with name, description, joints, category, available, path.
    """
    robots = []
    for r in list_robots(mode="sim"):
        name = r["name"]
        present = is_robot_asset_present(name)
        info = get_robot(name) or {}
        # Report where the asset is, never fetch one that is not here.
        path = resolve_model_path(name, allow_download=False) if present else None
        robots.append(
            {
                "name": name,
                "description": r.get("description", ""),
                "joints": r.get("joints"),
                "category": r.get("category", ""),
                "dir": info.get("asset", {}).get("dir", ""),
                "available": present,
                "path": str(path) if path else None,
            }
        )
    return robots
